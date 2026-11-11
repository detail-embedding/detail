"""Recording — the Python API boundary for one recording (Section 15).

The GUI, CLI, and scripts call these methods; nothing outside the API mutates
recording state or resolves internal paths (Tenet 2). ``run_*`` methods validate
everything before submitting (Tenet 10), return a run handle immediately, and
record the run in ``recording.yaml``.
"""

import contextlib
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ValidationError as PydanticValidationError


from detail.api._runner_modules import PIPELINE_RUNNERS as _PIPELINE_RUNNERS
from detail.core.dataclasses import (
    ComputeTarget,
    JobStatus,
    ProgressEvent,
    Prompt,
    PromptTarget,
    PromptTestJob,
    PromptTestResult,
    RenderPreviewJob,
    RenderPreviewResult,
    TextCue,
)
from detail.core.protocols import ModelNotReadyError
from detail.core.registries import STREAM_CONSUMERS, STREAM_IMPORTERS, TRANSFORMS
from detail.core.utils import active_polygon_at_frame, default_compute
from detail.core.visualize import bgr_to_rgb
from detail.executor import rundir
from detail.executor.base import RunHandle
from detail.executor.handles import ExecutorRunHandle
from detail.executor.local import LocalExecutor, cancel_run_dir
from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
from detail.importers import import_stream_importers
from detail.models.resolve import resolve_detection_model_auto
from detail.pipeline import prompt_test as _prompt_test_pipeline
from detail.pipeline import render_preview as _render_preview_pipeline
from detail.storage import clocks as clock_store
from detail.storage import experiment as experiment_store
from detail.storage import integrity
from detail.storage import read_cache
from detail.storage import recording as recording_store
from detail.storage import spatial as spatial_store
from detail.storage import stream_presets
from detail.storage import streams as streams_store
from detail.storage import video as video_store
from detail.storage import video_integrity
from detail.storage import video_roots
from detail.storage.video import read_frame_at
from detail.storage.zarr_io import PartZarrReader

if TYPE_CHECKING:
    from detail.api import previews


#: Processing modules this version no longer has, by the registered module whose run
#: namespace their output was written into. A run record outlives the module that wrote it
#: (``docs/removed_capabilities.md``), and deleting one still has to find its files: a
#: ``derive`` run wrote an ordinary ``emb_NN`` under ``embeddings/``, indexed in
#: ``embedding_runs``. Read only where a record is cleaned up — never to launch anything.
_REMOVED_MODULE_NAMESPACE = {"derive": "embedding"}


def _module_owning_output(module_id: str):
    """The registered module that knows where a ``module_id`` run's output lives.

    Raises:
        ValueError: ``module_id`` is neither registered nor a removed module whose
            namespace is known.
    """
    import_runner_modules(_PIPELINE_RUNNERS)
    module = PROCESSING_MODULES.get(_REMOVED_MODULE_NAMESPACE.get(module_id, module_id))
    if module is None:
        raise ValueError(
            f"unknown module_id {module_id!r}; available: {sorted(PROCESSING_MODULES)}"
        )
    return module


def _downstream_edges(module_id: str) -> list[tuple[str, str, str]]:
    """``(downstream_module, runs_key, ref_field)`` for every stage that builds on ``module_id``.

    Derived from each registered module's ``lineage_ref`` and ``aux_refs`` (§6.5) rather than
    listed by hand: the hand map held one downstream per stage, so the day a second stage
    (keypoints, §26) read tracking, deleting a tracking run would have orphaned its keypoint
    runs without a word. The registry is filled here first — nothing on the delete path
    imports the runner modules, and an empty registry would let a delete through.
    """
    import_runner_modules(_PIPELINE_RUNNERS)
    edges = {
        (m.module_id, m.runs_key, m.lineage_ref[1])
        for m in PROCESSING_MODULES.values()
        if m.lineage_ref is not None and m.lineage_ref[0] == module_id
    }
    edges |= {
        (m.module_id, m.runs_key, ref_field)
        for m in PROCESSING_MODULES.values()
        for aux_module, ref_field in getattr(m, "aux_refs", ())
        if aux_module == module_id
    }
    return sorted(edges)


#: The stages ``add_video_part`` reports, in the order it runs them. Named because the import
#: queue weighs each one's share of a part by name (`api/imports.py::_phase_spans`).
STAGE_TIMESTAMPS = "Reading timestamps"
STAGE_COPYING = "Copying"


def _size_label(done: int, total: int) -> str:
    """Human byte progress — GB for large files, MB otherwise."""
    if total >= 1 << 30:
        return f"{done / (1 << 30):.1f} / {total / (1 << 30):.1f} GB"
    return f"{done >> 20} / {total >> 20} MB"


def _relink_root(path: Path, source: dict) -> tuple[str, str, str | None]:
    """The video-root alias and relative path a relinked part should record.

    Returns ``(alias, rel, registered)``, where ``registered`` names an alias this call newly
    registered on this machine (else ``None``). In order:

    1. The part's current root, when it is registered here and holds ``path`` (a file that
       moved within its root).
    2. The part's current root, **registered here now**, when it is not registered on this
       machine and ``path`` ends in the recorded ``rel`` — the experiment came from another
       machine and the file is where the alias would have put it. Registering the alias rather
       than a new one is what makes one Locate fix every recording that shares it.
    3. Whatever an import would have chosen (``resolve_or_register_root``): the deepest root
       holding ``path``, else its folder under an auto alias. Picking a file is all a user
       does; the root is an internal detail, never a precondition.
    """
    resolved = path.resolve()
    registered = video_roots.list_video_roots()
    alias, rel = source.get("root"), source.get("rel")
    if alias in registered:
        with contextlib.suppress(ValueError):
            return (
                alias,
                resolved.relative_to(registered[alias].resolve()).as_posix(),
                None,
            )
    elif alias and rel:
        depth = len(Path(rel).parts)
        if len(resolved.parents) >= depth:
            implied = resolved.parents[depth - 1]
            if (implied / rel).resolve() == resolved:
                video_roots.register_video_root(alias, implied)
                return alias, rel, alias
    chosen = video_roots.resolve_or_register_root(resolved)
    root_dir = video_roots.video_root(chosen).resolve()
    new_alias = chosen if chosen not in registered else None
    return chosen, resolved.relative_to(root_dir).as_posix(), new_alias


class _RecordedRun(RunHandle):
    """Run handle that writes its final status back to a recording.yaml run list.

    One class for every processing module (§6.5) — parameterized by the module's
    ``runs_key`` rather than subclassed per stage.
    """

    def __init__(
        self,
        inner: ExecutorRunHandle,
        recording: "Recording",
        run_id: str,
        runs_key: str,
    ) -> None:
        self._inner = inner
        self._recording = recording
        self.id = run_id
        self._runs_key = runs_key

    @property
    def status(self) -> str:
        return self._inner.status

    @property
    def error(self) -> str | None:
        return self._inner.error

    def progress(self) -> Iterator[ProgressEvent]:
        return self._inner.progress()

    def wait(self) -> "_RecordedRun":
        """Drain the worker, then adopt its real status into ``recording.yaml``.

        **The adoption is unconditional**, because this is the only place it happens. Every
        downstream gate — tracking's validate, spec resolution, embedding, dataset export, and
        detection's refusal to start a second active job — reads the recording's own copy of
        the status, not ``job.json``. So a ``wait()`` that drains and then fails to adopt leaves
        the recording unusable *and* un-retryable, which is precisely how #289 turned one
        transient file error into a permanently stuck recording.

        The drain is I/O (it polls the envelope), so it can fail for reasons that say nothing
        about the job. Whatever it does, the status on disk is the truth and gets copied here —
        the same adoption ``Recording.cancel_run`` performs, and the reason a caller never has
        to reach for that after a failed wait.

        Fixed here rather than in ``RunOrchestrator``: the CLI, the GUI, distill and annotation
        all call ``wait()``, and each of them had its own way to skip it.
        """
        try:
            self._inner.wait()
        finally:
            self._adopt_status()
        return self

    def _adopt_status(self) -> None:
        """Copy the job's on-disk status into ``recording.yaml``. Never raises.

        A status we cannot read is one we cannot adopt, and there is nothing better to do than
        leave the record as it is — reconcile-on-open is the backstop for that. What must not
        happen is this failing *outward*, since it runs from a ``finally`` and would then
        replace the caller's real exception with a file-system one.
        """
        with contextlib.suppress(Exception):
            self._recording._set_run_status(self._runs_key, self.id, self._inner.status)

    def cancel(self) -> None:
        self._inner.cancel()


class Recording:
    def __init__(self, experiment_dir: Path | str, recording_id: str) -> None:
        self._experiment_dir = Path(experiment_dir)
        self.id = recording_id
        self.name = recording_store.read_recording(self.dir)["name"]

    @property
    def dir(self) -> Path:
        return experiment_store.recording_path(self._experiment_dir, self.id)

    @property
    def import_status(self) -> str:
        """``ready`` | ``queued`` | ``importing`` | ``failed`` | ``cancelled`` —
        defaults to ``ready`` for recordings created before the field existed (and
        for ones with no deferred import work)."""
        return recording_store.read_recording(self.dir).get("import_status", "ready")

    @property
    def video_available(self) -> bool:
        """True iff the recording has video parts **and** all their files are present
        on disk. False for a video-excluded export (entries + timestamps kept, the
        ``.mp4`` blobs absent) — preview surfaces check this to degrade gracefully
        instead of failing to decode."""
        files = (
            recording_store.read_recording(self.dir).get("video", {}).get("files", [])
        )
        return bool(files) and all(self._part_present(f) for f in files)

    def _part_present(self, part: dict) -> bool:
        """Whether one part's pixels are reachable on this machine.

        A reference-in-place part resolves through the machine-local video-root registry,
        which *raises* when the alias is not registered (a cross-machine export, a reset
        storage.yaml) — while a merely missing file just resolves to an absent path. Both
        mean "not reachable here", so both answer False: this is the presence gate, and
        every caller of :attr:`video_available` treats it as a question, not an assertion.
        The raise is left in place for the readers that actually need the pixels — they
        must fail loudly rather than silently skip a recording.
        """
        try:
            return recording_store.resolve_part_video_path(self.dir, part).is_file()
        except video_roots.MissingVideoRootError:
            return False

    def add_video_part(
        self,
        path: Path | str,
        storage: str = "default",
        video_root: str | None = None,
        progress: Callable[[str, float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Import a video file as the next part: register it with the recording,
        generate its timestamp map, and append it to ``recording.yaml`` (Section 3).

        The source is **copied in** verbatim, so the recording holds the footage at its
        original rate and resolution — a stage that wants fewer frames sets a
        ``every_n_frames`` on the run (§4), which needs no second copy. With
        ``storage="reference"`` the part is **referenced in place**: the video is *not*
        copied; the recording stores its location (a ``{root, rel}`` reference — the seam a
        remote backend plugs into) plus a content fingerprint, so a moved or changed file can
        be detected (Section 2). ``video_root`` is optional — when omitted, the video's own
        folder is remembered automatically (the caller references a path, never picks a
        location). The external source is only read, never modified. ``t_ns`` is each frame's
        time, global across parts (offset by the cumulative duration before it, Tenet 8).

        Raises:
            FileNotFoundError: If ``path`` is not a file.
            ValueError: If a reference import is missing/mismatched against its video root.
        """
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"video file not found: {path}")
        reference = storage == "reference"
        rel: str | None = None
        if reference:
            if video_root is None:
                # No location given (the GUI/CLI default): remember the video's own folder
                # automatically — the user references a path, never picks a location.
                video_root = video_roots.resolve_or_register_root(src)
            root_dir = video_roots.video_root(video_root)
            if root_dir is None:
                raise ValueError(
                    f"video root {video_root!r} is not registered; register it with "
                    f"`detail video-root set {video_root} <path>`"
                )
            try:
                rel = src.resolve().relative_to(root_dir.resolve()).as_posix()
            except ValueError:
                raise ValueError(
                    f"video {src} is not under video root {video_root!r} ({root_dir})"
                )
        data = recording_store.read_recording(self.dir)
        video = data.get("video") or {"storage": storage, "files": []}

        index = len(video["files"])
        part_name = f"part_{index + 1:03d}"
        probe = video_store.probe_video(src)
        fps = probe["fps"]
        # Integer ns from the parts' own maps, each re-based to start at 0 — the
        # offset is the only thing that places this part in the recording (Tenet 8).
        offset_ns = recording_store.video_end_ns(self.dir, data)
        dest = self.dir / "video" / f"{part_name}.mp4"
        on_timestamps = (
            (lambda f: progress(STAGE_TIMESTAMPS, f, ""))
            if progress is not None
            else None
        )
        try:
            if reference:
                # Reference in place: read the source's timestamps, never copy it.
                local_t_ns = video_store.generate_timestamps(
                    src, fps, progress=on_timestamps
                )
                eff_fps = fps
                eff_resolution = probe["resolution"]
            else:
                local_t_ns = video_store.generate_timestamps(
                    src, fps, progress=on_timestamps
                )
                on_bytes = (
                    (lambda d, t: progress(STAGE_COPYING, d / t, _size_label(d, t)))
                    if progress is not None
                    else None
                )
                video_store.copy_with_progress(
                    src, dest, progress=on_bytes, should_stop=should_stop
                )
                eff_fps = fps
                eff_resolution = probe["resolution"]
        except video_store.ImportStopped:
            dest.unlink(missing_ok=True)  # remove the partial part; no entry written
            raise
        t_ns = local_t_ns + offset_ns
        # Measured, not len/advertised-fps: the PTS map was demuxed precisely so this
        # number is real, and it sets every later part's offset (see offset_ns above).
        duration_s = video_store.part_duration_ns(local_t_ns, eff_fps) / (
            video_store.NS_PER_SECOND
        )

        ts_rel = f"video/timestamps/generated/{part_name}.csv"
        recording_store.write_timestamps_csv(self.dir / ts_rel, t_ns)

        if index == 0:
            video.update(
                fps_container=fps,
                fps_actual=eff_fps,
                fps_source="container",
                resolution=eff_resolution,
                total_duration_s=0.0,
            )
        entry = {
            "index": index,
            "id": part_name,  # stable stem for zarr/progress names (Section 3)
            "timestamps": {"generated": ts_rel, "user": None, "active": "generated"},
        }
        if reference:
            # No absolute path in the experiment file (Tenet 6): the location is a
            # {root, rel} reference resolved through the machine-local video-root
            # registry, plus a content fingerprint to detect a moved/changed file.
            entry["source"] = {"root": video_root, "rel": rel}
            entry["fingerprint"] = integrity.file_fingerprint(src)
            entry["frame_count"] = len(t_ns)
        else:
            entry["file"] = f"video/{part_name}.mp4"
            # Where this was copied *from*, in the same {root, rel} shape a referenced
            # part uses (Tenet 6 — the absolute path is composed at display time, never
            # stored). Provenance rather than a location: the tool owns the copy, so
            # nothing reads this to find pixels, but "which file was this?" still needs an
            # answer once the name has been edited away from the filename.
            #
            # Deliberately **not** ``source``: ``resolve_part_video_path`` keys on that to
            # decide where a part's frames live, so reusing it would send every read to
            # the external original instead of the copy — deleting the original would then
            # break runs we hold the footage for.
            origin_root = video_roots.resolve_or_register_root(src)
            origin_dir = video_roots.video_root(origin_root)
            entry["origin"] = {
                "root": origin_root,
                "rel": src.resolve().relative_to(origin_dir.resolve()).as_posix(),
            }
        video["files"].append(entry)
        video["total_duration_s"] += duration_s
        data["video"] = video
        recording_store.write_recording(self.dir, data)

    def verify_videos(self) -> list[video_integrity.VideoIssue]:
        """Check every video part is present and (for referenced parts) unchanged (Section 2).

        Returns the list of integrity issues (empty means all good) — the same check the run
        pre-flight gate enforces, exposed so a user can check without launching a run.
        """
        data = recording_store.read_recording(self.dir)
        return video_integrity.validate_source_videos(self.dir, data)

    def relink_video_part(self, part_index: int, new_path: Path | str) -> str | None:
        """Point a referenced video part at a moved/renamed file, verified by fingerprint.

        Only a reference part can be relinked (a copied part's file is managed by the tool). The
        new file's content fingerprint must match what was recorded — a mismatch means the
        content *changed*, not just moved, and is refused (remove and re-add the recording
        instead). The new path need not be under a registered video root: one is registered
        for it, the way an import does. When the part's own root is not registered on this
        machine (an experiment made elsewhere) and the file sits where that root would put it,
        the root itself is registered, so every other recording referencing it resolves too.

        Returns:
            The video-root alias this call registered on this machine, or ``None`` if it
            used one that was already registered.

        Raises:
            FileNotFoundError: ``new_path`` is not a file.
            IndexError: ``part_index`` is out of range.
            ValueError: the part is copied (not referenced) or the content differs.
        """
        new = Path(new_path)
        if not new.is_file():
            raise FileNotFoundError(f"video file not found: {new_path}")
        data = recording_store.read_recording(self.dir)
        files = data.get("video", {}).get("files", [])
        if not (0 <= part_index < len(files)):
            raise IndexError(f"part_index {part_index} out of range [0, {len(files)})")
        part = files[part_index]
        if part.get("source") is None:
            raise ValueError(
                f"part {part_index} is a copied part, not referenced — nothing to relink"
            )
        recorded = part.get("fingerprint") or {}
        current = integrity.file_fingerprint(new)
        if current["size"] != recorded.get("size") or current[
            "sha256_ends"
        ] != recorded.get("sha256_ends"):
            raise ValueError(
                f"{new} has different content than the recorded video — it is a changed or "
                "different video, not a move; remove and re-add the recording to use it"
            )
        alias, rel, registered = _relink_root(new, part["source"])
        part["source"] = {"root": alias, "rel": rel}
        part["fingerprint"] = current  # a moved/renamed file may have a fresh mtime
        recording_store.write_recording(self.dir, data)
        return registered

    def define_arena(
        self,
        name: str,
        keyframes: list[tuple[int, list[tuple[float, float]]]],
    ) -> str:
        """Define a named arena polygon (Section 10).

        Each keyframe is ``(frame_idx, [(x, y), ...])`` with recording-global
        ``frame_idx`` (0 = first frame of part_001) and normalized vertices in
        ``[0, 1]``. All keyframes must have the same vertex count.

        Returns:
            The auto-assigned stable arena id (e.g. ``"arena_001"``) — this is
            what ``TrackingJob.arena_ids`` refers to.

        Raises:
            ValueError: If keyframes are invalid (see
                ``storage.spatial.write_arena_keyframes``) or the name is empty
                or already in use within this recording.
        """
        arena_id = spatial_store.next_arena_id(self.dir)
        n_vertices = spatial_store.write_arena_keyframes(self.dir, arena_id, keyframes)
        spatial_store.append_arena_to_recording_yaml(
            self.dir, arena_id, name, n_vertices
        )
        return arena_id

    def arena_dependents(self, arena_id: str) -> list[str]:
        """Tracking run ids that used ``arena_id`` — editing or removing the arena
        would make their recorded region inconsistent, so it is blocked while any
        exist (delete those runs first)."""
        data = recording_store.read_recording(self.dir)
        return [
            t["id"]
            for t in data.get("tracking_runs", [])
            if arena_id in (t.get("arena_ids") or [])
        ]

    def arena_keyframes(
        self, arena_id: str
    ) -> list[tuple[int, list[tuple[float, float]]]]:
        """The arena's polygon keyframes as ``[(frame_idx, [(x, y), ...]), ...]``
        (normalized vertices), e.g. to reload it into the editor.

        Raises:
            ValueError: If no arena with this id exists in the recording.
        """
        arenas = {a["id"]: a for a in self.arenas}
        if arena_id not in arenas:
            raise ValueError(f"unknown arena_id {arena_id!r}")
        # Cached on the CSV's own mtime. The spatial map reads one of these **per recording**
        # every time a result page opens, which at a hundred recordings is a hundred CSV parses
        # for a polygon that changes only when somebody edits the arena. The keyframes are
        # rebuilt into fresh lists below, so a hit hands nobody a shared mutable.
        path = spatial_store.arena_keyframes_path(self.dir, arena_id)
        pk = read_cache.cached_read(
            ("arena_keyframes", str(self.dir), arena_id),
            path,
            lambda: spatial_store.read_arena_keyframes(
                self.dir, arena_id, arenas[arena_id]["n_vertices"]
            ),
        )
        return [
            (int(frame), [(float(x), float(y)) for x, y in verts])
            for frame, verts in pk.keyframes
        ]

    def clear_arenas(self) -> None:
        """Remove all arenas (keyframes + recording.yaml entries) so the recording's
        single arena can be redrawn. Already-computed runs are unaffected.

        Raises:
            ValueError: If a tracking run used one of the arenas (see
                ``arena_dependents``) — delete that run first.
        """
        arenas = recording_store.read_recording(self.dir).get("arenas", [])
        blocked = {
            a["id"]: deps for a in arenas if (deps := self.arena_dependents(a["id"]))
        }
        if blocked:
            detail = "; ".join(f"{aid} ← {deps}" for aid, deps in blocked.items())
            raise ValueError(
                f"cannot edit/clear arena(s) used by tracking runs: {detail} "
                "— delete those runs first"
            )
        for arena in arenas:
            spatial_store.delete_arena(self.dir, arena["id"])

    def delete_arena(self, arena_id: str) -> None:
        """Remove a single arena (keyframes + recording.yaml entry). Already-computed
        runs are unaffected — the polygon was applied at run time.

        Raises:
            ValueError: If a tracking run used this arena (see ``arena_dependents``)
                — delete that run first.
        """
        deps = self.arena_dependents(arena_id)
        if deps:
            raise ValueError(
                f"cannot delete arena {arena_id!r} used by tracking runs: {deps} "
                "— delete those runs first"
            )
        spatial_store.delete_arena(self.dir, arena_id)

    # --- ROIs and POIs (Section 10, analysis-time) -------------------------- #
    # Regions of interest (polygons) and points of interest (single points) are
    # analysis-time definitions: unlike arenas they do NOT affect detection or
    # tracking, so there is no run-lock and delete is unconditional. Each may
    # carry an optional ``active`` interval (whole video when ``None``).

    def _spatial_entry(self, kind: str, manifest_key: str, def_id: str) -> dict:
        entries = {
            e["id"]: e
            for e in recording_store.read_recording(self.dir).get(manifest_key, [])
        }
        if def_id not in entries:
            raise ValueError(f"unknown {kind} id {def_id!r}")
        return entries[def_id]

    def _spatial_keyframes(
        self, kind: str, manifest_key: str, def_id: str
    ) -> list[tuple[int, list[tuple[float, float]]]]:
        entry = self._spatial_entry(kind, manifest_key, def_id)
        pk = spatial_store.read_keyframes(self.dir, kind, def_id, entry["n_vertices"])
        return [
            (int(f), [(float(x), float(y)) for x, y in verts])
            for f, verts in pk.keyframes
        ]

    def _spatial_at_frame(
        self, kind: str, manifest_key: str, def_id: str, frame_idx: int
    ) -> np.ndarray | None:
        entry = self._spatial_entry(kind, manifest_key, def_id)
        pk = spatial_store.read_keyframes(self.dir, kind, def_id, entry["n_vertices"])
        active = entry.get("active")
        interval = (active["start"], active["end"]) if active else None
        return active_polygon_at_frame(pk, interval, frame_idx)

    def define_roi(
        self,
        name: str,
        keyframes: list[tuple[int, list[tuple[float, float]]]],
        active: tuple[int, int] | None = None,
    ) -> str:
        """Define a named region-of-interest polygon (Section 10).

        ROIs are **analysis-time only** — they do not affect detection or
        tracking. Keyframes match ``define_arena`` (recording-global ``frame_idx``,
        normalized ``[0, 1]`` vertices, >= 3 per keyframe). ``active`` is an
        optional inclusive ``(start, end)`` recording-global frame interval the ROI
        is present for; ``None`` = whole video.

        Returns:
            The auto-assigned stable id (e.g. ``"roi_001"``).

        Raises:
            ValueError: If keyframes/active are invalid, or the name is empty or
                already used within this recording.
        """
        roi_id = spatial_store.next_id(self.dir, "roi")
        n = spatial_store.write_keyframes(self.dir, "roi", roi_id, keyframes)
        spatial_store.append_to_manifest(self.dir, "roi", roi_id, name, n, active)
        return roi_id

    def roi_keyframes(self, roi_id: str) -> list[tuple[int, list[tuple[float, float]]]]:
        """The ROI's polygon keyframes (to reload it into the editor).

        Raises:
            ValueError: If no ROI with this id exists in the recording.
        """
        return self._spatial_keyframes("roi", "rois", roi_id)

    def roi_at_frame(self, roi_id: str, frame_idx: int) -> np.ndarray | None:
        """The ROI polygon at ``frame_idx`` as an ``(n_vertices, 2)`` array, or
        ``None`` if the ROI is not active at that frame (Section 10)."""
        return self._spatial_at_frame("roi", "rois", roi_id, frame_idx)

    def delete_roi(self, roi_id: str) -> None:
        """Remove an ROI (keyframes + recording.yaml entry). Unconditional —
        nothing consumes ROIs yet."""
        spatial_store.delete_def(self.dir, "roi", roi_id)

    def define_point(
        self,
        name: str,
        keyframes: list[tuple[int, list[tuple[float, float]]]],
        active: tuple[int, int] | None = None,
    ) -> str:
        """Define a named point of interest (Section 10) — a landmark point
        (lever, port, hopper).

        Each keyframe carries exactly **one** normalized vertex; a single keyframe
        is a static point, more keyframes move it (e.g. a panning camera).
        Analysis-time only (does not affect detection/tracking). ``active`` as in
        ``define_roi``. Distinct from pose keypoints (user-placed, not model-derived).

        Returns:
            The auto-assigned stable id (e.g. ``"point_001"``).
        """
        point_id = spatial_store.next_id(self.dir, "point")
        n = spatial_store.write_keyframes(self.dir, "point", point_id, keyframes)
        spatial_store.append_to_manifest(self.dir, "point", point_id, name, n, active)
        return point_id

    def point_keyframes(
        self, point_id: str
    ) -> list[tuple[int, list[tuple[float, float]]]]:
        """The POI's keyframes (one vertex each).

        Raises:
            ValueError: If no POI with this id exists in the recording.
        """
        return self._spatial_keyframes("point", "points", point_id)

    def point_at_frame(self, point_id: str, frame_idx: int) -> np.ndarray | None:
        """The point as a ``(1, 2)`` array at ``frame_idx``, or ``None`` if the POI
        is not active at that frame (Section 10)."""
        return self._spatial_at_frame("point", "points", point_id, frame_idx)

    def delete_point(self, point_id: str) -> None:
        """Remove a POI (keyframes + recording.yaml entry). Unconditional."""
        spatial_store.delete_def(self.dir, "point", point_id)

    def set_rois(self, defs: list[dict]) -> list[str]:
        """Replace **all** ROIs in one transaction — the multi-region manager's "Save all"
        (Section 10). Each entry is ``{"name", "keyframes", "active"}`` (``active``
        optional). Existing ROIs are removed and fresh ids assigned; returns the new ids.

        Raises:
            ValueError: If any def is invalid (empty/duplicate name, bad keyframes/active).
        """
        return spatial_store.replace_defs(self.dir, "roi", defs)

    def set_points(self, defs: list[dict]) -> list[str]:
        """Replace **all** POIs in one transaction — the manager's "Save all" (Section 10).
        Each entry is ``{"name", "keyframes", "active"}``; each keyframe carries one
        vertex. Returns the new ids."""
        return spatial_store.replace_defs(self.dir, "point", defs)

    def export_spatial_csv(self, path: Path | str) -> None:
        """Write this recording's whole spatial setup to a round-trippable CSV (Section 10)
        — the field of view, the distance calibration, and the ROIs + POIs — for sharing or
        importing onto another recording.

        One file per recording rather than one per kind; the import side is per kind, so a
        single export can feed any of them.
        """
        spatial_store.export_spatial(self.dir, path)

    def import_spatial_csv(
        self, path: Path | str, kinds: tuple[str, ...] = ("roi", "point")
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Take the ``kinds`` rows of an exported spatial CSV onto this recording (§10).

        Goes through :meth:`receive_spatial`, so a setup **pulled** from a file obeys the
        same rules as one **pushed** from a recording: nothing is replaced, a name already
        here is refused rather than suffixed ``(2)``, keyframe motion is flattened, and an
        active window does not travel.

        Returns:
            ``(added, flattened, refused)``.
        """
        by_kind = {}
        for d in spatial_store.load_spatial_csv(path):
            by_kind.setdefault(d["kind"], []).append(d)
        added, flattened, refused = [], [], []
        for kind in kinds:
            if not by_kind.get(kind):
                continue
            a, f, r = self.receive_spatial(by_kind[kind], kind)
            added += a
            flattened += f
            refused += r
        if not added and not refused:
            raise ValueError(f"{path} carries no {' or '.join(kinds)}")
        return added, flattened, refused

    def import_spatial_from(
        self, source: "Recording", kinds: tuple[str, ...] = ("roi", "point")
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Take another recording's ROIs/POIs onto this one (§10) — the *pull* twin of
        ``Experiment.propagate_rois`` / ``propagate_points``, and the same rules.

        Returns:
            ``(added, flattened, refused)``.
        """
        added, flattened, refused = [], [], []
        for kind in kinds:
            defs = spatial_store.read_defs(source.dir, kind)
            if not defs:
                continue
            a, f, r = self.receive_spatial(defs, kind)
            added += a
            flattened += f
            refused += r
        if not added and not refused:
            raise ValueError(
                f"recording {source.id!r} has no {' or '.join(kinds)} to copy"
            )
        return added, flattened, refused

    #: Kinds whose collisions are judged per definition rather than per kind. Mirrors
    #: `gui.arena.SpatialKind.collision_grain` so the API and CLI need not import the GUI
    #: to know the rule. A field of view and a calibration are refused whole — two arenas
    #: break the batch tracking picker, which intersects arena *ids* across recordings, and
    #: there is one pixel scale. Regions and points are sets, so only the taken names are
    #: refused; failing a whole set on one clash would kill "pull one more zone from a
    #: template", which is what `recording import-spatial` has always been for.
    _PER_DEFINITION_KINDS = ("roi", "point")

    def field_of_view_refusal(self) -> str | None:
        """Why this recording cannot receive a field of view, or ``None``.

        One sentence for both directions. It lived only on the push side, so pulling a
        field of view onto a locked recording said "already has one" and stopped, while
        pushing the identical setup named the tracking runs you would have to delete
        first — same situation, two qualities of advice.

        It carries no "here" or "there": the pull direction reads it about the recording
        you are on and the push direction reads it about a row in a list of sixty others,
        so either word is wrong half the time.
        """
        if not self.arenas:
            return None
        blocking = sorted(
            {
                run_id
                for arena in self.arenas
                for run_id in self.arena_dependents(arena["id"])
            }
        )
        if blocking:
            return (
                "already has a field of view, used by tracking "
                f"{'runs' if len(blocking) > 1 else 'run'} {', '.join(blocking)} — "
                "delete those runs, then that field of view"
            )
        return "already has a field of view — delete the existing one first"

    def receive_spatial(
        self, defs: list[dict], kind: str
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Take ``defs`` (already parsed, from a recording or a CSV) onto this recording.

        The one door for a setup arriving from elsewhere, whichever direction it was
        started from and whichever source it came from. Applies
        ``storage.spatial.prepare_for_arrival``'s timeline policy, then refuses by name for
        the set-shaped kinds — nothing is ever replaced.

        Returns:
            ``(added, flattened, refused)`` — names written, names whose keyframe motion
            was dropped, and ``[(name, reason), ...]`` left behind.

        Raises:
            ValueError: nothing in ``defs`` could be taken.
        """
        prepared, flattened, refused = spatial_store.prepare_for_arrival(defs)
        if kind in self._PER_DEFINITION_KINDS:
            taken = {e["name"] for e in spatial_store.list_defs(self.dir, kind)}
            keep = []
            for d in prepared:
                if d["name"] in taken:
                    refused.append(
                        (
                            d["name"],
                            f"already exists on {self.name} — rename or delete the "
                            "existing one first; nothing is overwritten",
                        )
                    )
                else:
                    keep.append(d)
            prepared = keep
        # Never raises for "nothing to take". This is called once per kind, so raising
        # here let a single windowed region abort a transfer whose points were perfectly
        # transferable — the caller could see neither. Refusals are data; the caller
        # accumulates them across kinds and decides.
        if not prepared:
            return [], flattened, refused
        spatial_store.import_spatial(self.dir, prepared)
        return [d["name"] for d in prepared], flattened, refused

    def propagate_spatial_from(
        self, source: "Recording", kind: str, names: list[str] | None = None
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Take a copy of ``source``'s ``kind`` definitions (Section 10).

        A thin read in front of :meth:`receive_spatial`, which owns the policy.

        Returns:
            ``(added, flattened, refused)``.
        """
        defs = spatial_store.read_defs(source.dir, kind)
        if names is not None:
            defs = [d for d in defs if d["name"] in set(names)]
        return self.receive_spatial(defs, kind)

    # --- field-of-view carry-over (§10) ------------------------------------- #
    # Recordings off one rig share a field of view, so it carries over like the ROIs/POIs
    # do — but by **replacement**, not appending: "make this recording match that one"
    # means one field of view, not a second one. Replacing goes through ``clear_arenas``,
    # so a field of view a tracking run depends on refuses rather than silently
    # invalidating that run's recorded region (DESIGN_PLAN §4016).

    def import_field_of_view_from(
        self, source: "Recording"
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Take ``source``'s field of view onto this recording (§10).

        **Never replaces.** A recording that already has one keeps it; delete it here first
        (and the tracking runs holding it, which ``arena_dependents`` names). This used to
        ``clear_arenas()`` and copy over the top after a Yes/No — the same operation the
        *push* direction refuses outright, which is how one setup arriving from elsewhere
        came to have two contradictory rules depending on which way you started it.

        Returns:
            ``(added, flattened, refused)``.
        """
        refusal = self.field_of_view_refusal()
        if refusal is not None:
            raise ValueError(f"recording {self.id!r} {refusal}")
        defs = spatial_store.read_defs(source.dir, "arena")
        if not defs:
            raise ValueError(f"recording {source.id!r} has no field of view to copy")
        return self.receive_spatial(defs, "arena")

    def import_field_of_view_csv(
        self, path: Path | str
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Take the field of view out of an exported spatial CSV (§10). Never replaces —
        see :meth:`import_field_of_view_from`.

        Returns:
            ``(added, flattened, refused)``.
        """
        defs = [d for d in spatial_store.load_spatial_csv(path) if d["kind"] == "arena"]
        if not defs:
            raise ValueError(f"{path} carries no field of view")
        refusal = self.field_of_view_refusal()
        if refusal is not None:
            raise ValueError(f"recording {self.id!r} {refusal}")
        return self.receive_spatial(defs, "arena")

    # --- distance-calibration carry-over (§10) ------------------------------ #
    # Singular, so it always replaces. Unlike the field of view there is no run lock:
    # a completed analysis run already wrote its CSVs, so only future runs see the change.

    def import_calibration_from(self, source: "Recording") -> dict:
        """Take ``source``'s distance calibration onto this recording (§10).

        Never replaces: there is one pixel scale, so a recording that already has one keeps
        it. Nothing to flatten for the timeline — a calibration is two points, a distance
        and the resolution it was measured at, with no frame index anywhere. Refused onto a
        video of a different shape, where no one pixel scale is right.

        Raises:
            ValueError: the source has none, this recording already has one, or its video's
                aspect ratio differs from the one the calibration was measured on.
        """
        if self.distance_calibration is not None:
            raise ValueError(
                f"recording {self.id!r} already has a distance calibration — delete it "
                "here first; nothing is overwritten"
            )
        return spatial_store.copy_calibration(
            source.dir, self.dir, self.video_resolution
        )

    def import_calibration_csv(self, path: Path | str) -> dict:
        """Take the distance calibration out of an exported spatial CSV (§10). Never
        replaces — see :meth:`import_calibration_from`.

        Raises:
            FileNotFoundError | ValueError: a missing/malformed CSV, one carrying no
                calibration, a recording that already has one, or one whose video is a
                different shape from the one the calibration was measured on.
        """
        block = spatial_store.load_calibration_csv(path)
        if block is None:
            raise ValueError(f"{path} carries no distance calibration")
        if self.distance_calibration is not None:
            raise ValueError(
                f"recording {self.id!r} already has a distance calibration — delete it "
                "here first; nothing is overwritten"
            )
        return spatial_store.write_calibration_block(
            self.dir, block, self.video_resolution
        )

    # --- distance calibration (§10) ----------------------------------------- #
    @property
    def distance_calibration(self) -> dict | None:
        """The recording's distance-calibration block (Section 10), or ``None`` if not set.
        Fields: ``pixels_per_mm``, ``ref_resolution`` ``[W, H]``, and (for a two-point pick)
        the normalized ``points`` and ``distance_mm``."""
        return spatial_store.read_calibration(self.dir)

    @property
    def pixels_per_mm(self) -> float | None:
        """``pixels_per_mm`` at the recording's current video resolution (rescaled from the
        calibration's own resolution), or ``None`` if uncalibrated.

        Raises:
            ValueError: the calibration cannot describe this video — see
                :meth:`calibration_refusal`, which a display should ask first.
        """
        cal = self.distance_calibration
        if cal is None:
            return None
        width, height = self.video_resolution
        return spatial_store.pixels_per_mm_at(cal, width, height)

    def calibration_refusal(self) -> str | None:
        """Why this recording's distance calibration gives no scale for its video, or
        ``None`` (also when it has no calibration or no video).

        Reachable although every carry-over refuses it: the video can be replaced after
        the calibration was set, and a calibration copied before the refusal existed is
        still on disk.
        """
        return self.calibration_refusal_for(self.distance_calibration)

    def calibration_refusal_for(self, calibration: dict | None) -> str | None:
        """Why ``calibration`` gives no scale for this recording's video, or ``None``
        (also when either is missing). The question a copy asks of a target before it is
        offered: a calibration belongs to a rig, and a video of another shape is another
        rig."""
        res = self.video_resolution
        if calibration is None or res is None:
            return None
        return spatial_store.calibration_aspect_refusal(calibration, *res)

    def set_distance_calibration(
        self, points: list[tuple[float, float]], distance_mm: float
    ) -> dict:
        """Calibrate from two normalized points a known real distance (mm) apart — the GUI
        two-point pick (Section 10). Uses the current video resolution as the reference.

        Raises:
            ValueError: if there is no video, the points coincide, or ``distance_mm <= 0``.
        """
        return spatial_store.write_calibration_from_points(
            self.dir, points, distance_mm, self._require_resolution()
        )

    def set_pixels_per_mm(self, pixels_per_mm: float) -> dict:
        """Calibrate directly from a ``pixels_per_mm`` scalar (the CLI/scripted path, no
        two-point pick), measured at the current video resolution (Section 10).

        Raises:
            ValueError: if there is no video or ``pixels_per_mm <= 0``.
        """
        return spatial_store.write_calibration_scalar(
            self.dir, pixels_per_mm, self._require_resolution()
        )

    def clear_distance_calibration(self) -> None:
        """Remove the distance calibration (idempotent)."""
        spatial_store.delete_calibration(self.dir)

    def _require_resolution(self) -> tuple[int, int]:
        res = self.video_resolution
        if res is None:
            raise ValueError("cannot calibrate a recording with no imported video")
        return res

    # --- location export (per-recording ROI/POI spatial features) ------------- #
    def _require_complete_tracking(self, tracking_run_id: str) -> dict:
        rec = {r["id"]: r for r in self.tracking_runs}.get(tracking_run_id)
        if rec is None:
            raise ValueError(
                f"tracking run {tracking_run_id!r} not found in recording {self.id!r}"
            )
        if rec.get("status") != "complete":
            raise ValueError(
                f"tracking run {tracking_run_id!r} is {rec.get('status')!r}, not complete"
            )
        return rec

    def location_tables(self, tracking_run_id: str):
        """Per-frame, per-track spatial features for a completed tracking run —
        bbox/centroids/areas + position relative to this recording's ROIs/POIs.

        Returns an ``analysis.locations.LocationTables`` (compute only, no write) —
        the reuse hook for a preview or a future analysis module. Read-only; the
        compute is pure (no executor, no run record).

        Raises:
            ValueError: If the tracking run is missing or not complete.
        """
        from detail.analysis import locations as _locations

        self._require_complete_tracking(tracking_run_id)
        return _locations.compute_location_tables(self, tracking_run_id)

    def export_locations(self, tracking_run_id: str, out_dir: Path | str) -> Path:
        """Compute + write the three location CSVs (``tracks`` / ``roi_relations`` /
        ``point_relations``) + a ``locations.yaml`` manifest into ``out_dir``;
        returns ``out_dir``. Tables keyed by ``t_ns`` (joins with streams)."""
        from detail.storage import location_export

        tables = self.location_tables(tracking_run_id)
        return location_export.write_location_tables(out_dir, tables)

    def backfill_kinematics(
        self, embed_run_id: str, *, progress=None, overwrite: bool = False
    ) -> int:
        """Write the §14 ``kinematics/`` sibling for an embedding run that has none.

        A run made before kinematics existed has no per-frame geometry, so nothing downstream
        can say *where* a behavior happened. This regenerates it from the detection zarr — the
        masks and boxes are still there — without a GPU, a model, or a video decode. Returns the
        number of parts written.

        ``progress`` takes the standard ``(stage, fraction, detail)`` callback; on a long
        recording this is minutes, and a silent long op is indistinguishable from a hang.
        Already-written parts are skipped unless ``overwrite``, so an interrupted run resumes.
        """
        from detail.pipeline import kinematics_backfill

        return kinematics_backfill.backfill_run(
            self.dir, embed_run_id, progress=progress, overwrite=overwrite
        )

    def missing_previews(self, tracking_run_ids: list[str] | None = None) -> list:
        """The preview builds this recording is missing on this machine (§3 *Video preview
        cache*) — each part's proxy, and each complete tracking run's masks. One stat per part
        once the cache knows the file; empty when everything is built."""
        from detail.api import previews

        return previews.recording_tasks(
            self.dir, self.id, tracking_run_ids=tracking_run_ids
        )

    def build_preview(
        self,
        progress: Callable[[str, float, str], None] | None = None,
        *,
        tracking_run_ids: list[str] | None = None,
        masks: bool = True,
    ) -> "previews.PreviewBuildReport":
        """Build this recording's display-only previews on this machine; returns a
        ``previews.PreviewBuildReport`` (``built``, and ``failed`` — each failed build with
        why; one failure does not stop the rest).

        Each part's proxy (480x270 at most, 15 fps) and, with ``masks``, each complete
        tracking run's masks at proxy size (``tracking_run_ids`` limits which). They live in
        the machine-local preview cache, never in the experiment, and no pipeline stage reads
        them (§3). Minutes per hour of video, so ``progress`` takes the standard
        ``(stage, fraction, detail)`` callback; a callback that raises stops the build and the
        part in hand is discarded.
        """
        from detail.api import previews

        tasks = self.missing_previews(tracking_run_ids)
        if not masks:
            tasks = [task for task in tasks if task.kind == previews.PROXY]
        return previews.build_tasks(
            self._experiment_dir, tasks, {self.id: self.name}, progress
        )

    def kinematics_missing_parts(self, embed_run_id: str) -> list[Path]:
        """Embedding parts of this run with no kinematics sibling — empty when it is complete."""
        from detail.pipeline import kinematics_backfill

        return kinematics_backfill.missing_parts(self.dir, embed_run_id)

    def kinematics(self, embed_run_id: str, track_id: int = 0) -> dict | None:
        """Where the subject was on each embedded frame (§14), or ``None`` if never captured.

        ``{frame_idx, t_ns, cx_norm, cy_norm, mask_area_norm, bbox_norm}``, all normalized to the
        frame so two recordings of different resolutions are comparable. The same rows, in the
        same order, as the embedding run's own — without reading the embedding column beside it.

        ``None`` for a run made before this was captured; :meth:`backfill_kinematics` fills it in.
        """
        from detail.api.run_preview import load_track_kinematics

        return load_track_kinematics(self.dir, embed_run_id, track_id)

    # --- streams (Section 3) ------------------------------------------------ #
    def add_stream(
        self,
        name: str,
        raw_path: Path | str,
        importer: str,
        aux_paths: dict[str, Path | str] | None = None,
        covered_behaviors: list[str] | None = None,
        origin: str | None = None,
        config: dict | None = None,
        offset_ns: int = 0,
    ) -> None:
        """Import a time-series stream and record it in recording.yaml (Section 3).

        Runs the named ``StreamImporter`` over ``raw_path`` (plus any
        ``aux_paths`` it declares) to produce canonical rows in stream-local time,
        copies the source files verbatim into ``streams/<name>/raw/``, writes the
        canonical CSV, and appends the manifest.

        Args:
            name: Unique, path-safe stream name (also the directory name).
            raw_path: The primary data file to import.
            importer: A registered importer id (e.g. ``"medpc"``).
            aux_paths: Extra files keyed by the importer's ``aux_inputs`` keys
                (e.g. ``{"program": "...SelfAdmin.MPC"}`` for MED-PC).
            covered_behaviors: For a ``behavior`` stream, the full scored-behavior
                vocabulary (declares behaviors that were scored but never occurred,
                so they still count as covered for eval eligibility). Must be a
                superset of the labels present. Defaults to the labels present.
            origin: For a ``behavior`` stream, ``"ground_truth"`` (a human scored
                the video) or ``"predicted"`` (a method's output, re-ingested).
                Defaults to the file's own ``origin`` column when it has one — the
                tool's own exporters write it — and to ``"ground_truth"`` otherwise.
                A predicted stream is never read as ground truth (§19), so marking
                it is what keeps a method from scoring against itself.
            config: The importer's own parameters, matching its ``config_model``
                (e.g. ``CsvStreamConfig``'s time column + channels). ``None`` means
                the importer's own defaults, which is what every importer did before
                it had a config. Validated against the model up front (Tenet 10).
            offset_ns: The stream's alignment offset, set at import instead of as a
                separate edit afterwards — ``video_t_ns = stream_t_ns + offset_ns``,
                positive if the stream began after the video. Only meaningful on an
                external clock; a ``video_frames`` / ``video_seconds`` stream is
                anchored at import and a non-zero offset for one is refused rather
                than stored, because it would be silently ignored on read.

        Raises:
            ValueError: Unknown importer, invalid/duplicate name, a missing
                required aux input, a ``config`` its model rejects, a non-zero
                ``offset_ns`` on a video-anchored stream, ``covered_behaviors`` not
                covering a present label, or an ``origin`` outside the allowed set.
            FileNotFoundError: If ``raw_path`` or any aux path is not a file.
        """
        import_stream_importers()
        streams_store.validate_stream_name(name)
        importer_cls = STREAM_IMPORTERS.get(importer)
        if importer_cls is None:
            raise ValueError(
                f"unknown importer {importer!r}; available: {sorted(STREAM_IMPORTERS)}"
            )
        config = self._validated_stream_config(importer, importer_cls, config)
        raw_path = Path(raw_path)
        if not raw_path.is_file():
            raise FileNotFoundError(f"stream data file not found: {raw_path}")
        resolved_aux = {k: Path(v) for k, v in (aux_paths or {}).items()}
        instance = importer_cls()
        missing = [a.key for a in instance.aux_inputs if a.key not in resolved_aux]
        if missing:
            raise ValueError(
                f"importer {importer!r} requires aux file(s) {missing} "
                f"(pass them in aux_paths)"
            )
        for key, path in resolved_aux.items():
            if not path.is_file():
                raise FileNotFoundError(f"aux file {key!r} not found: {path}")
        data = recording_store.read_recording(self.dir)
        if any(s.get("name") == name for s in data.get("streams", [])):
            raise ValueError(f"stream name {name!r} already exists in recording")

        sdir = streams_store.stream_dir(self.dir, name)
        raw_dir = sdir / streams_store.RAW_DIRNAME
        canonical_dir = sdir / streams_store.CANONICAL_DIRNAME
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_entries = [self._copy_raw(raw_dir, raw_path, 0, "data")]
        for i, (key, path) in enumerate(resolved_aux.items(), start=1):
            raw_entries.append(self._copy_raw(raw_dir, path, i, key))

        result = instance.import_stream(raw_path, canonical_dir, resolved_aux, config)
        # Only a frame/time-clocked importer needs the timestamp map; a t_ns-native
        # stream (e.g. MED-PC) skips it, so a recording without video still imports.
        needs_map = bool(result.frame_columns or result.time_columns)
        frame_t_ns = (
            self.frame_timestamps() if needs_map else np.empty(0, dtype=np.int64)
        )
        rows = clock_store.resolve_native_columns(
            result.columns,
            result.rows,
            result.frame_columns,
            result.time_columns,
            frame_t_ns,
        )
        canonical_name = streams_store.write_canonical_csv(
            canonical_dir, result.columns, rows
        )
        canonical_rel = (
            f"{streams_store.STREAMS_DIRNAME}/{name}/"
            f"{streams_store.CANONICAL_DIRNAME}/{canonical_name}"
        )
        dtypes = result.dtypes or streams_store.default_dtypes(result.columns)
        metadata = dict(result.metadata or {})
        if covered_behaviors is not None:
            if "label" not in result.columns:
                raise ValueError(
                    "covered_behaviors only applies to a labeled (behavior) stream"
                )
            present = {r[result.columns.index("label")] for r in result.rows}
            missing = present - set(covered_behaviors)
            if missing:
                raise ValueError(
                    "covered_behaviors must include every labeled behavior; missing "
                    f"{sorted(missing)}"
                )
            metadata["covered_behaviors"] = sorted(set(covered_behaviors))
        resolved_origin = self._resolve_stream_origin(instance, result, origin)
        # Checked against the *resolved* clock, not the importer's declared one: a CSV read
        # on the frame kind reports `video_frames` from a `stream_seconds` importer, so the
        # declared clock would wrongly accept an offset that read time then ignores.
        resolved_clock = result.clock or instance.clock
        if offset_ns and not clock_store.needs_alignment(resolved_clock):
            raise ValueError(
                f"stream {name!r} is on the {resolved_clock!r} clock, which is already the "
                "video's own, so an alignment offset would be ignored on read; import it "
                "with offset_ns=0"
            )
        manifest = {
            "name": name,
            "type": result.type,
            "shape": result.shape or streams_store.stream_shape({"type": result.type}),
            "role": instance.role,
            "clock": resolved_clock,
            "columns": result.columns,
            "dtypes": dtypes,
            "importer": importer,
            "alignment": {"offset_ns": int(offset_ns), "active": "offset"},
            "raw": raw_entries,
            "files": [{"index": 0, "file": canonical_rel}],
            "metadata": metadata,
        }
        if result.sample_rate_hz is not None:
            manifest["sample_rate_hz"] = result.sample_rate_hz
        if resolved_origin is not None:
            manifest["origin"] = resolved_origin
        streams_store.append_stream_to_recording_yaml(self.dir, manifest)

    @staticmethod
    def _validated_stream_config(
        importer: str, importer_cls, config: dict | None
    ) -> dict | None:
        """``config`` checked against the importer's ``config_model``, or ``None``.

        Validated here rather than inside the importer so a bad mapping is refused before
        anything is copied or written (Tenet 10), and so every caller — API, CLI, GUI,
        bundle manifest — gets the same message from one place.

        Raises:
            ValueError: If the importer declares no ``config_model`` but a config was
                given, or the model rejects it.
        """
        if config is None:
            return None
        model = getattr(importer_cls, "config_model", None)
        if model is None:
            raise ValueError(
                f"importer {importer!r} takes no config, but one was given "
                f"({sorted(config)})"
            )
        try:
            return model.model_validate(config).model_dump()
        except PydanticValidationError as exc:
            raise ValueError(
                f"invalid config for importer {importer!r}: {exc}"
            ) from None

    @staticmethod
    def _resolve_stream_origin(importer, result, origin: str | None) -> str | None:
        """Where this stream's labels came from, or ``None`` for a non-behavior stream.

        Priority: the caller's explicit argument, then the imported file's own ``origin``
        column (the tool's own exporters write one, so a prediction self-identifies), then
        ``ground_truth``. The explicit argument wins so a user can correct a file that
        misdeclares itself — or mark a prediction that came from somewhere else entirely.

        Raises:
            ValueError: ``origin`` is outside the allowed set, or was passed for a stream
                that carries no labels (where it would mean nothing).
        """
        if importer.role != "behavior":
            if origin is not None:
                raise ValueError(
                    "origin only applies to a labeled (behavior) stream; "
                    f"importer {importer.importer_id!r} has role {importer.role!r}"
                )
            return None
        resolved = origin or (result.metadata or {}).get("origin")
        resolved = str(resolved or streams_store.DEFAULT_BEHAVIOR_ORIGIN)
        if resolved not in streams_store.BEHAVIOR_ORIGINS:
            raise ValueError(
                f"origin must be one of {sorted(streams_store.BEHAVIOR_ORIGINS)}, got "
                f"{resolved!r}"
            )
        return resolved

    def list_stream_presets(self) -> list[dict]:
        """The **experiment's** saved importer setups (Section 3).

        On ``Recording`` as well as ``Experiment`` because a preset is used from a
        recording — the surface that imports a stream should not have to reach past itself
        to find out what setups exist.
        """
        return stream_presets.list_presets(self._experiment_dir)

    def get_stream_preset(self, name: str) -> dict:
        """Raises: ValueError if no preset named ``name`` exists."""
        return stream_presets.get_preset(self._experiment_dir, name)

    def save_stream_preset(
        self,
        name: str,
        importer: str,
        config: dict | None,
        *,
        offset_ns: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """Save an importer + config on the experiment, for its other recordings.

        Raises:
            ValueError: An invalid name, or the name exists and ``overwrite`` is False.
        """
        stream_presets.save_preset(
            self._experiment_dir,
            name,
            importer,
            config,
            offset_ns=offset_ns,
            overwrite=overwrite,
        )

    def add_stream_from_preset(
        self,
        name: str,
        raw_path: Path | str,
        preset: str,
        *,
        offset_ns: int | None = None,
        aux_paths: dict[str, Path | str] | None = None,
        origin: str | None = None,
    ) -> None:
        """Import a stream using a saved preset's importer + config (Section 3).

        The preset supplies the ``importer`` and its ``config``; everything that belongs to
        *this* recording — the file, the stream name, the alignment offset — stays an
        argument, because those are what differ between recordings even when the mapping
        does not.

        Args:
            offset_ns: This recording's alignment offset. ``None`` uses the preset's own,
                if it declared one, else 0. Alignment is per recording, so a preset rarely
                carries one (see ``storage/stream_presets.py``).

        Raises:
            ValueError: Unknown preset, or any ``add_stream`` error.
            FileNotFoundError: If ``raw_path`` is not a file.
        """
        entry = stream_presets.get_preset(self._experiment_dir, preset)
        resolved_offset = entry.get("offset_ns", 0) if offset_ns is None else offset_ns
        self.add_stream(
            name,
            raw_path=raw_path,
            importer=entry["importer"],
            aux_paths=aux_paths,
            origin=origin,
            config=entry.get("config") or None,
            offset_ns=int(resolved_offset),
        )

    def _copy_raw(self, raw_dir: Path, src: Path, index: int, role: str) -> dict:
        """Copy a source file into the stream's ``raw/`` dir; return its manifest entry."""
        dest = raw_dir / src.name
        shutil.copy2(src, dest)
        return {
            "index": index,
            "file": dest.relative_to(self.dir).as_posix(),
            "role": role,
        }

    @property
    def streams(self) -> list[dict]:
        """Stream manifest entries from recording.yaml (empty list if none)."""
        return streams_store.list_streams(self.dir)

    def streams_by_role(self, role: str) -> list[dict]:
        """Stream manifest entries with the given ``role`` (Section 3).

        The discovery surface a role-consuming module reads through — e.g. the
        behavior eval finds recordings carrying a ``behavior``-role stream.
        """
        return streams_store.streams_by_role(self.dir, role)

    @property
    def is_annotated(self) -> bool:
        """Whether the recording carries behavior ground-truth labels (Section 3).

        Presence of any **ground-truth** ``behavior``-role stream means the recording
        is annotated; per-behavior eligibility uses ``is_annotated_for`` (an ethogram
        scores only some behaviors). This gates a recording for a behavior-eval run.

        A stream carrying a method's own re-ingested output (``origin: predicted``)
        does not annotate anything — counting it here would let a run be gated in on
        the very predictions it is about to be scored against (§19).
        """
        from detail.analysis.behavior_labels import ground_truth_streams

        return bool(ground_truth_streams(self))

    def _behavior_stream(self) -> dict | None:
        """The recording's ``behavior``-role stream manifest, or ``None`` if it carries none.

        Routed through ``behavior_labels.resolve_behavior_stream`` so the implicit choice is
        made in exactly one place: with several streams and no name it raises rather than
        taking whichever was added first (see that function for why).
        """
        from detail.analysis.behavior_labels import resolve_behavior_stream

        name = resolve_behavior_stream(self, None)
        if name is None:
            return None
        return next((s for s in self.streams if s.get("name") == name), None)

    def covered_behaviors(self, name: str | None = None) -> list[str]:
        """Behaviors the annotation *scored* (Section 3) — the eligibility vocabulary.

        Reads the manifest's ``covered_behaviors`` (importers declare it, so a
        scored-but-never-occurred behavior still counts); falls back to the distinct
        labels present for a legacy stream that predates the field. ``name`` selects
        a stream; default is the first ``behavior`` stream. Empty if none.
        """
        manifest = (
            next((s for s in self.streams if s.get("name") == name), None)
            if name is not None
            else self._behavior_stream()
        )
        if manifest is None:
            return []
        covered = (manifest.get("metadata") or {}).get("covered_behaviors")
        if covered is not None:
            return list(covered)
        return sorted({r[2] for r in self.stream_events(manifest["name"])})

    def behavior_types(self, name: str | None = None) -> dict[str, str]:
        """Per-behavior ``point``/``state`` type (Section 3).

        Reads the manifest's ``behavior_types`` (importers derive it — BORIS from its
        type column, the CSV from interval durations); falls back to inferring from
        the stream's own durations for a legacy stream. ``name`` selects a stream;
        default is the first ``behavior`` stream. Empty if none.
        """
        manifest = (
            next((s for s in self.streams if s.get("name") == name), None)
            if name is not None
            else self._behavior_stream()
        )
        if manifest is None:
            return {}
        types = (manifest.get("metadata") or {}).get("behavior_types")
        if types is not None:
            return dict(types)
        has_duration: dict[str, bool] = {}
        for row in self.stream_events(manifest["name"]):
            label, start, end = row[2], row[0], row[1]
            has_duration[label] = has_duration.get(label, False) or (end > start)
        return {
            label: ("state" if dur else "point") for label, dur in has_duration.items()
        }

    def is_annotated_for(self, behavior: str, name: str | None = None) -> bool:
        """Whether the recording's annotation scored ``behavior`` (Section 3).

        True iff ``behavior`` is in the stream's covered vocabulary — so a query
        "miss" of it counts as a real false negative, not an unscored behavior.

        A **predicted** stream never makes a recording eligible, whether it was named
        or resolved (§19): a method scored against its own output measures nothing,
        and the failure is silent because the numbers come out looking excellent.
        Returns ``False`` rather than raising — this is an eligibility question, and
        "not annotated for this" is the honest answer to it.
        """
        if name is not None:
            entry = next((s for s in self.streams if s.get("name") == name), None)
            if entry is not None and streams_store.is_predicted(entry):
                return False
        return behavior in self.covered_behaviors(name)

    def stream_events(self, name: str) -> list[tuple]:
        """The stream's canonical rows, typed per its manifest ``dtypes`` (Section 3).

        For a MED-PC ``events`` stream each row is ``(t_ns, code, label)``; for a
        ``behavior`` interval stream ``(t_ns, end_t_ns, label, subject, start_frame,
        end_frame)``; for a continuous stream ``(t_ns, *channels)``.

        ``t_ns`` is **stream-local**: the manifest's ``alignment.offset_ns`` is not applied
        here. For a video-anchored stream (``video_frames`` / ``video_seconds``, e.g.
        behavior labels) that is already video time, because the offset of such a stream is
        definitionally 0. For an external-clock stream it is the device's own time, and
        ``stream_preview(name).events(aligned=True)`` is the accessor that maps it to video
        time. The distinction matters to anything *computing* an offset — the sync-to-video
        helper derives ``frame_t_ns - anchor_t_ns`` from these rows, and pre-aligned rows
        would count the offset twice.

        Raises:
            FileNotFoundError: If the stream's canonical file is absent.
        """
        manifest = next((s for s in self.streams if s.get("name") == name), None)
        dtypes = manifest.get("dtypes") if manifest else None
        _columns, rows = streams_store.read_canonical(self.dir, name, dtypes=dtypes)
        return rows

    def stream_preview(self, name: str):
        """A read-only accessor for previewing a stream (Section 3): its events in
        stream-local and video-aligned time, grouped for a timeline raster.

        Raises:
            ValueError: If no stream with this name exists in the recording.
        """
        from detail.api.run_preview import StreamPreview

        return StreamPreview(self.dir, name)

    def set_stream_offset(self, name: str, offset_ns: int) -> None:
        """Set the stream's constant alignment offset (signed ns; Section 3).

        ``video_t_ns = stream_t_ns + offset_ns`` — positive if the stream began
        after the video, negative if before. Applied at read time; the canonical
        data is never rewritten.

        Raises:
            ValueError: If no stream with this name exists.
        """
        streams_store.set_stream_offset(self.dir, name, offset_ns)

    def stream_dependents(self, name: str) -> list[str]:
        """Runs that build on this stream and would be orphaned by deleting it.

        Aggregates every consumer registered for the stream's ``role`` in
        ``STREAM_CONSUMERS``. Empty today — no stage consumes streams yet — but a
        future consumer (the behavior eval) registers there without editing this
        method, keeping the deletion strategy (gui_design §1.1) uniform.
        """
        manifest = next((s for s in self.streams if s.get("name") == name), None)
        if manifest is None:
            return []
        dependents: list[str] = []
        for consumer in STREAM_CONSUMERS.get(manifest.get("role"), []):
            dependents.extend(consumer(self.dir, name))
        return dependents

    def delete_stream(self, name: str) -> None:
        """Remove a stream's directory + its recording.yaml entry (idempotent)."""
        streams_store.delete_stream(self.dir, name)

    def set_name(self, name: str) -> None:
        """Rename the recording (id/dir unchanged) — the label every surface displays.

        A name a user typed that is already taken is **refused**, not silently suffixed:
        the tool quietly renaming what someone deliberately typed is the worse failure.
        (The import path, where the *tool* proposes filenames and collisions are routine,
        suffixes instead — ``Experiment.add_recording``.)

        Raises:
            ValueError: If ``name`` is empty, or another recording already has it.
        """
        # Local import: api.experiment imports Recording at module level, so this would
        # be circular at the top.
        from detail.api.experiment import Experiment

        experiment = Experiment.open(self._experiment_dir)
        holder = next(
            (
                rid
                for rid, existing in experiment.recording_names().items()
                if existing == name and rid != self.id
            ),
            None,
        )
        if holder is not None:
            raise ValueError(
                f"another recording is already named {name!r} (id {holder}); "
                "names identify recordings to you, so they have to be unique"
            )
        recording_store.set_name(self.dir, name)
        self.name = name

    @property
    def video_sources(self) -> list[str]:
        """Where each video part's footage came from — one entry per part, ``""`` when
        unknown (a part imported before origins were recorded; honest, not a guess).

        One accessor because the user's question is the same ("which file is this?"). Both
        kinds store ``{root, rel}`` and are composed into an absolute path here, at display
        time — what they *mean* differs: a **referenced** part's ``source`` is the video's
        live location (relinking rewrites it, so this follows the file), while a **copied**
        part's ``origin`` is where it came from (the tool holds the footage; the original
        may since have moved, and nothing breaks if it has).
        """
        parts = (
            recording_store.read_recording(self.dir).get("video", {}).get("files", [])
        )
        out = []
        for part in parts:
            ref = part.get("source") or part.get("origin")
            if not ref:
                # A part imported before origins were recorded. `origin_name` covers the
                # brief window when only the filename was kept.
                out.append(part.get("origin_name", ""))
                continue
            try:
                out.append(str(video_roots.resolve_video_ref(ref)))
            except video_roots.MissingVideoRootError:
                # The root isn't registered on this machine (an experiment carried over
                # from another box). Name the file, since the path would be a guess;
                # whether the video is reachable is verify_videos' question, not this one.
                out.append(Path(ref.get("rel", "")).name)
        return out

    def reorder_video_parts(self, new_order: list[int]) -> None:
        """Reorder the video parts and re-offset global timestamps (Tenet 8).

        Part order is the recording's time reference, so this applies only before
        any processing run exists. ``new_order`` is the current part indices in
        the desired sequence.

        Raises:
            ValueError: If a processing run already exists (it was computed
                against the current order), or ``new_order`` is invalid.
        """
        data = recording_store.read_recording(self.dir)
        n_runs = sum(
            len(data.get(key) or [])
            for key in (
                "detection_runs",
                "tracking_runs",
                "embedding_runs",
            )
        )
        if n_runs:
            raise ValueError(
                f"cannot reorder video parts: {n_runs} processing run(s) exist "
                "and were computed against the current order — delete them first"
            )
        recording_store.reorder_video_parts(self.dir, new_order)

    @property
    def video_parts(self) -> list[dict]:
        """Video part entries from recording.yaml — ``[{index, file, timestamps}, ...]``.

        Read accessor so callers never read recording.yaml directly (Tenet 2)."""
        return (
            recording_store.read_recording(self.dir).get("video", {}).get("files", [])
        )

    @property
    def video_fps(self) -> float | None:
        """The recording's frame rate, or None if no video has been imported.

        The container's advertised rate, read from ``recording.yaml`` — cheap enough for a
        form to consult per keystroke. It is a *display* number: the frame-exact reference
        is the PTS timestamp map, and a stride is an exact integer either way (§4, Tenet 8).
        """
        video = recording_store.read_recording(self.dir).get("video", {})
        fps = video.get("fps_actual")
        return float(fps) if fps else None

    @property
    def video_resolution(self) -> tuple[int, int] | None:
        """The recording's source video resolution ``(width, height)``, or None if
        no video has been imported."""
        video = recording_store.read_recording(self.dir).get("video", {})
        res = video.get("resolution")
        return (int(res[0]), int(res[1])) if res else None

    @property
    def part_frame_counts(self) -> list[int]:
        """Frame count of each video part, from its active timestamp map.

        Lets a caller map a recording-global frame index to ``(part_index,
        local_idx)`` — e.g. the arena-draw canvas's single continuous frame
        slider (§10)."""
        data = recording_store.read_recording(self.dir)
        files = data.get("video", {}).get("files", [])
        return [
            len(recording_store.load_active_timestamps(self.dir, data, i))
            for i in range(len(files))
        ]

    def frame_t_ns(self, global_frame_idx: int) -> int:
        """Video ``t_ns`` of a recording-global frame index (Section 3, Tenet 8).

        Maps the continuous frame index (0 = first frame of part_001) to its
        ``(part, local)`` position via the per-part active timestamp maps and
        returns that frame's global ``t_ns``. Powers stream→video alignment
        (the "sync to video frame" offset = ``frame_t_ns(frame) - anchor_t_ns``).

        Raises:
            IndexError: If ``global_frame_idx`` is negative or past the last frame.
        """
        if global_frame_idx < 0:
            raise IndexError(f"global_frame_idx must be >= 0, got {global_frame_idx}")
        data = recording_store.read_recording(self.dir)
        files = data.get("video", {}).get("files", [])
        remaining = global_frame_idx
        for part_index in range(len(files)):
            tmap = recording_store.load_active_timestamps(self.dir, data, part_index)
            if remaining < len(tmap):
                return tmap.t_ns(remaining)
            remaining -= len(tmap)
        raise IndexError(
            f"global_frame_idx {global_frame_idx} is past the end of the recording"
        )

    def frame_timestamps(self) -> np.ndarray:
        """Concatenated global video ``t_ns`` per recording-global frame index.

        One int64 value per frame across all parts (Tenet 8) — the timeline axis
        for previews, and the inverse of ``frame_t_ns`` for a whole recording
        without re-reading the maps per lookup. Empty when no video is imported.
        """
        data = recording_store.read_recording(self.dir)
        files = data.get("video", {}).get("files", [])
        if not files:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [
                recording_store.load_active_timestamps(self.dir, data, i).values
                for i in range(len(files))
            ]
        )

    @property
    def arenas(self) -> list[dict]:
        """Arena entries from recording.yaml — ``[{id, name, n_vertices}, ...]``."""
        return recording_store.read_recording(self.dir).get("arenas", [])

    @property
    def rois(self) -> list[dict]:
        """ROI entries — ``[{id, name, n_vertices, active}, ...]`` (``active`` is
        ``{start, end}`` or ``None`` = whole video)."""
        return recording_store.read_recording(self.dir).get("rois", [])

    @property
    def points(self) -> list[dict]:
        """POI entries — ``[{id, name, n_vertices, active}, ...]`` (``n_vertices``
        is 1; ``active`` is ``{start, end}`` or ``None`` = whole video)."""
        return recording_store.read_recording(self.dir).get("points", [])

    def _cursor_for(self, part_index: int, path, timestamps):
        """The reader for one part, keeping its place between calls.

        Playback asks for the *next* frame forty times a second, and
        :func:`~detail.storage.video.read_frame_at` re-opens and re-seeks for every one of them —
        over an order of magnitude slower than a cursor on a 1080p part, so 1x playback
        showed a fraction of its frames and 4x was unreachable.

        **One cursor, not one per part.** A cursor holds an open container, and this object is
        held for a dialog's lifetime by each arena and stream surface; keeping one per part would
        make the number of open file handles a property of how many parts a recording happens to
        have. Surfaces scrub within a part and cross a boundary rarely, so switching parts closes
        the old one and opens the new — the cost of a boundary crossing is exactly the seek it
        replaced.

        Keyed on the resolved path as well as the index, so a relinked video (``RelinkVideoDialog``)
        is not served from a cursor still holding the file it replaced.

        Held for this object's lifetime, but the *file* is not: the cursor closes its container
        after ``FrameCursor.IDLE_CLOSE_S`` without a request, so a user can move the video in a
        file manager while the page showing it stays open — on Windows an open handle forbids it.
        """
        from detail.storage.video import FrameCursor

        key = (int(part_index), str(path))
        if getattr(self, "_frame_cursor_key", None) != key:
            existing = getattr(self, "_frame_cursor", None)
            if existing is not None:
                existing.close()
            self._frame_cursor = FrameCursor(path, timestamps=timestamps)
            self._frame_cursor_key = key
        return self._frame_cursor

    def get_frame(self, part_index: int, frame_idx: int) -> np.ndarray:
        """Decode a single RGB frame from a video part — inspection helper.

        Intended for preview / inspection (e.g. arena drawing backdrop, GUI
        thumbnails). Frame seeking is permitted here under Tenet 11's preview
        exemption. Seeks through the part's timestamp map, so the frame returned
        is the ``frame_idx``-th in presentation order.

        Returns:
            ``(H, W, 3) uint8`` RGB array.

        Raises:
            IndexError: If ``part_index`` is out of range, or if the requested
                ``frame_idx`` cannot be decoded (e.g. past end-of-file).
        """
        data = recording_store.read_recording(self.dir)
        files = data.get("video", {}).get("files", [])
        if not (0 <= part_index < len(files)):
            raise IndexError(f"part_index {part_index} out of range [0, {len(files)})")
        if frame_idx < 0:
            raise IndexError(f"frame_idx must be >= 0, got {frame_idx}")
        path = recording_store.resolve_part_video_path(self.dir, files[part_index])
        # Cached: a thumbnail grid calls this per tile, and the map is both the seek
        # index and immutable once written.
        timestamps = recording_store.load_active_timestamps_cached(
            self.dir, data, part_index
        ).values
        decoded = self._cursor_for(part_index, path, timestamps).frame(frame_idx)
        if decoded is None:
            raise IndexError(
                f"could not decode frame {frame_idx} from part {part_index} ({path.name})"
            )
        _actual, frame_bgr = decoded
        return bgr_to_rgb(frame_bgr)  # cv2 returns BGR; API surface is RGB

    @property
    def detection_runs(self) -> list[dict]:
        return recording_store.read_recording(self.dir).get("detection_runs", [])

    def detection_preview(self, run_id: str):
        """A read-only per-frame overlay accessor for a completed detection run
        (decode + boxes), for in-GUI preview without writing an annotated video.

        Raises:
            ValueError: If no detection run with this id exists in the recording.
        """
        from detail.api.run_preview import DetectionPreview

        return DetectionPreview(self.dir, run_id)

    @property
    def tracking_runs(self) -> list[dict]:
        return recording_store.read_recording(self.dir).get("tracking_runs", [])

    def tracking_preview(self, run_id: str):
        """A read-only per-frame overlay accessor for a completed tracking run
        (decode + boxes colored by track id), for in-GUI preview.

        Raises:
            ValueError: If no tracking run with this id exists in the recording.
        """
        from detail.api.run_preview import TrackingPreview

        return TrackingPreview(self.dir, run_id)

    @property
    def embedding_runs(self) -> list[dict]:
        return recording_store.read_recording(self.dir).get("embedding_runs", [])

    def run_dependents(self, module_id: str, run_id: str) -> list[tuple[str, str]]:
        """Runs in this recording that build on ``run_id`` and would be orphaned by
        deleting it — as ``(module_id, downstream_run_id)`` pairs."""
        data = recording_store.read_recording(self.dir)
        return [
            (downstream_module, entry["id"])
            for downstream_module, downstream_key, ref_field in _downstream_edges(
                module_id
            )
            for entry in data.get(downstream_key, [])
            if entry.get(ref_field) == run_id
        ]

    def delete_run(self, module_id: str, run_id: str) -> None:
        """Remove a per-recording run's output directory + its recording.yaml entry.

        The caller must ensure nothing depends on it (see ``run_dependents``).
        Idempotent: a missing directory or entry is skipped.
        """
        module = _module_owning_output(module_id)
        out_dir = self.dir / module.output_path(run_id).rstrip("/")
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        recording_store.remove_run_entry(self.dir, module.runs_key, run_id)

    def cancel_run(self, module_id: str, run_id: str) -> str:
        """Stop a per-recording run: kill its worker (graceful-then-force, via the
        run dir — works even after the GUI restarted) and record its **real** on-disk
        status in recording.yaml. Returns that status — usually ``cancelled``, but a
        graceful stop keeps ``partial`` and a worker that already finished keeps
        ``complete`` (so a done run isn't relabeled). Idempotent."""
        module = _module_owning_output(module_id)
        out_dir = self.dir / module.output_path(run_id).rstrip("/")
        if out_dir.is_dir():
            cancel_run_dir(out_dir)
            status = rundir.read_status(out_dir).value
        else:
            status = "cancelled"  # nothing ran (no run dir) → treat as cancelled
        recording_store.set_run_status(self.dir, module.runs_key, run_id, status)
        return status

    def run(self, module_id: str, config: dict) -> RunHandle:
        """Run a processing module (§6.5): validate, submit, record, return a handle.

        The single backbone behind every ``run_*`` stage method. The module
        (detection / tracking / embedding, or a future stage)
        owns validation, job construction, and the recording.yaml record shape.

        Raises:
            ValueError: For an unknown ``module_id`` or any failed pre-run
                validation.
            ModelNotReadyError: If a model-backed stage's weights/credentials
                are not in place.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        module = PROCESSING_MODULES.get(module_id)
        if module is None:
            raise ValueError(
                f"unknown module_id {module_id!r}; available: "
                f"{sorted(PROCESSING_MODULES)}"
            )
        data = recording_store.read_recording(self.dir)
        # Resolve a batch <upstream>_spec to this recording's own run id before validate
        # (§6.5). A no-op for the explicit-id path (CLI/typed wrappers) or a module
        # with no lineage parent.
        config = module.resolve_upstream_spec(self.dir, data, config)
        config = module.validate(self.dir, data, config)

        # Reserve the id + record atomically (under the recording lock) so concurrent runs on
        # this recording — e.g. several embeddings pinned to different GPUs — never collide on the
        # id or lose-update the record. The record is committed as PENDING, then submitted, then
        # advanced to its real status; a submit failure marks it failed rather than orphaning it.
        run_id = self._reserve_run(
            module.runs_key,
            module.id_prefix,
            lambda rid: module.run_record(rid, config, JobStatus.PENDING.value),
        )
        executor = LocalExecutor(self._experiment_dir, runner_modules=_PIPELINE_RUNNERS)
        try:
            job = module.build_job(self.id, run_id, config)
            executor.submit(job)
        except Exception:
            self._set_run_status(module.runs_key, run_id, JobStatus.FAILED.value)
            raise
        self._set_run_status(module.runs_key, run_id, executor.status(run_id).value)
        return _RecordedRun(
            ExecutorRunHandle(executor, run_id), self, run_id, module.runs_key
        )

    def run_render_preview(
        self,
        tracking_run_id: str,
        track_ids: list[int],
        renderer_spec: list[dict],
        n_sample_frames: int,
        save_as: str | None = None,
        seed: int | None = None,
    ) -> list[RenderPreviewResult]:
        """Sample N detections from the requested tracks and render each (Section 14).

        Synchronous: blocks while N samples are decoded + rendered. Not an
        executor job — no zarr, no recovery, no recording.yaml entry. ``save_as``
        writes source + rendered JPGs and ``preview.json`` to
        ``embeddings/render_previews/<save_as>/`` for inspection in any image
        viewer.

        Raises:
            ValueError: For any failed pre-run validation.
        """
        data = recording_store.read_recording(self.dir)
        self._validate_render_preview(
            data, tracking_run_id, track_ids, renderer_spec, n_sample_frames
        )
        job = RenderPreviewJob(
            tracking_run_id=tracking_run_id,
            track_ids=list(track_ids),
            renderer_spec=renderer_spec,
            n_sample_frames=n_sample_frames,
        )
        return _render_preview_pipeline.run_render_preview(
            self.dir, job, save_as=save_as, seed=seed
        )

    def run_embedding(
        self,
        tracking_run_id: str,
        track_ids: list[int],
        model_id: str | None = None,
        renderer_spec: list[dict] | None = None,
        batch_size: int = 32,
        compute: ComputeTarget | None = None,
        model_variant: str | None = None,
        constructor: str | None = None,
        readout: str | None = None,
        input_size: int | None = None,
        every_n_frames: int | None = None,
        *,
        align: str = "none",
        keypoint_run_id: str | None = None,
        heading: str | None = None,
        max_gap_s: float = 0.5,
        mirror_hold_s: float = 1.0,
        facing_camera_ratio: float = 0.5,
    ) -> RunHandle:
        """Embed the selected tracks through the renderer + model (Section 14).

        ``constructor`` is the name the ``renderer_spec`` was resolved from; passing it records the
        constructor on the run so a constructor-axis comparison can read it (the spec alone does not
        name it). ``renderer_spec`` ``None`` resolves the constructor per recording in ``validate``,
        which is the route an **aligned** run takes (the aligned canvas is sized from the turned
        masks and the alignment step is inserted there). ``readout`` selects the feature read-out
        and ``input_size`` the model input resolution (both recorded for the model / resolution
        comparison axes). ``every_n_frames`` omitted inherits the tracking run's cadence.

        §14 *Egocentric alignment*: ``align`` ``rotate`` / ``mirror`` turns each subject image so
        the animal faces right, reading the heading from ``keypoint_run_id`` (a keypoint track run on
        the same tracking run) through the saved heading definition ``heading``; ``max_gap_s``,
        ``mirror_hold_s`` and ``facing_camera_ratio`` are its settings and part of the run's
        identity. ``model_id`` omitted runs the configured default (``EmbeddingConfig``), and
        ``renderer_spec`` and ``constructor`` both omitted render the default constructor.
        Thin wrapper over ``run``."""
        from detail.api import models as api_models
        from detail.configs.embedding import EmbeddingConfig

        model_id = model_id or EmbeddingConfig.model_fields["model_id"].default

        return self.run(
            "embedding",
            {
                "tracking_run_id": tracking_run_id,
                "track_ids": list(track_ids),
                "model_id": model_id,
                "model_variant": model_variant,
                "renderer_spec": renderer_spec,
                "constructor": constructor,
                "readout": readout,
                "input_size": input_size,
                "batch_size": batch_size,
                "every_n_frames": every_n_frames,
                "align": align,
                "keypoint_run_id": keypoint_run_id,
                "heading": heading,
                "max_gap_s": max_gap_s,
                "mirror_hold_s": mirror_hold_s,
                "facing_camera_ratio": facing_camera_ratio,
                "compute": compute
                or default_compute(api_models.supported_backends(model_id)),
            },
        )

    def track_ids(self, tracking_run_id: str) -> list[int]:
        """Sorted track ids a completed tracking run produced (from its
        ``tracks.zarr``). ``[]`` when the run has no tracks summary yet.

        A completed run's ``tracks.zarr`` is immutable, so the read is cached against
        its mtime — the GUI reads this once per selected recording on every selection
        change, and over many recordings the repeated zarr opens are the select-all lag.
        """
        tracks_path = self.dir / "tracking" / tracking_run_id / "tracks.zarr"
        if not tracks_path.exists():
            return []
        return list(
            read_cache.cached_read(
                ("track_ids", str(tracks_path)),
                tracks_path,
                lambda: sorted(
                    int(t) for t in PartZarrReader(tracks_path).read()["track_id"]
                ),
            )
        )

    def embedding_preview(self, run_id: str):
        """A read-only accessor for a completed embedding run (Section 14): run
        metadata, per-track coverage, a 2-D PCA of the embeddings, and the frame +
        model input behind any point.

        Raises:
            ValueError: If no embedding run with this id exists in the recording.
        """
        from detail.api.run_preview import EmbeddingPreview

        return EmbeddingPreview(self.dir, run_id)

    def input_image_preview(
        self,
        tracking_run_id: str,
        renderer_spec: list[dict],
        *,
        alignment: dict | None = None,
    ):
        """A read-only scrubber accessor: per source frame, the original frame and
        the model-input image a constructor produces for a track (Section 14).

        ``alignment`` (§14) is the resolved alignment a run would record — ``align``,
        ``keypoint_run_id``, ``heading_definition``, ``heading_cutoff``, ``max_gap_s``,
        ``mirror_hold_s``, ``facing_camera_ratio`` — so the preview turns each input the way
        the run will, and shows no input where the run would embed no row.

        Raises:
            ValueError: If no tracking run with this id exists in the recording.
        """
        from detail.api.run_preview import InputImagePreview

        return InputImagePreview(
            self.dir, tracking_run_id, renderer_spec, alignment=alignment
        )

    def resolve_constructor(
        self,
        constructor: str,
        tracking_run_id: str,
        track_ids: list[int],
        model_id: str | None = None,
        resize: int | None = None,
        input_size: int | None = None,
        canvas_px: int | None = None,
    ) -> list[dict]:
        """Resolve a named input-image constructor to a concrete renderer_spec for
        this recording (Section 14), computing the shared canvas from this recording's
        masks when the constructor needs it — or taking ``canvas_px`` as given, which is
        how an aligned preview passes the canvas sized from the turned masks (§14). The same resolution an embedding run
        applies — use it to preview a constructor's inputs before launching.

        ``model_id`` supplies the default resize target (the model's input size) so a
        preview shows exactly the image the run will embed; ``input_size`` (the model
        resolution knob) overrides that default, and ``resize`` overrides both.

        Raises:
            ValueError: For an unknown constructor, or tracks with no detections.
        """
        from detail.pipeline.embedding import resolve_renderer_spec

        return resolve_renderer_spec(
            self.dir,
            tracking_run_id,
            list(track_ids),
            constructor,
            model_id=model_id,
            resize=resize,
            input_size=input_size,
            canvas_px=canvas_px,
        )

    def run_prompt_test(
        self,
        model_id: str,
        prompts: list[Prompt],
        n_sample_frames: int,
        compute: ComputeTarget,
        save_as: str | None = None,
        seed: int | None = None,
        model_variant: str | None = None,
        display_classes: list[str] | None = None,
        min_conf: float = 0.0,
        on_result=None,
    ) -> list[PromptTestResult]:
        """Preview a detection model on M sampled frames and return per-frame results (Section 8).

        Open-vocab models (SAM3) compare the N candidate ``prompts``, one
        result column each. A trained/imported closed-set model takes no prompt —
        it previews its own classes; ``display_classes`` picks which (default: all),
        one result column per class.

        The model runs wide open, so every returned detection is a proposal with its own
        ``confidence`` — filter the results at whatever threshold you want to look at.
        ``min_conf`` applies only to the JPGs ``save_as`` writes, which can't be
        re-thresholded once they exist.

        Synchronous: blocks until the predictions complete. Not an executor job — no
        zarr, no recovery, no recording.yaml entry. ``save_as`` writes overlay JPGs and
        ``test.json`` to ``detections/prompt_tests/<save_as>/`` for inspection.

        Raises:
            ValueError: For any failed pre-run validation.
            ModelNotReadyError: If the model's weights / credentials are not in place.
        """
        import_runner_modules(_PIPELINE_RUNNERS)
        data = recording_store.read_recording(self.dir)
        closed_set, model_classes = self._validate_prompt_test(
            data,
            model_id,
            prompts,
            n_sample_frames,
            compute,
            model_variant,
            display_classes,
        )
        if closed_set:
            chosen = list(display_classes) if display_classes else model_classes
            job_prompts = [
                Prompt(targets=[PromptTarget(class_name=c, cue=TextCue(c))])
                for c in chosen
            ]
        else:
            job_prompts = list(prompts)
        job = PromptTestJob(
            recording_id=self.id,
            model_id=model_id,
            prompts=job_prompts,
            n_sample_frames=n_sample_frames,
            compute=compute,
            model_variant=model_variant,
        )
        return _prompt_test_pipeline.run_prompt_test(
            self.dir,
            job,
            save_as=save_as,
            seed=seed,
            min_conf=min_conf,
            on_result=on_result,
        )

    def run_detection(
        self,
        model_id: str,
        prompt: Prompt,
        compute: ComputeTarget,
        conf_threshold: float = 0.5,
        confirm_fn=None,
        model_variant: str | None = None,
        batch_size: int | None = None,
        every_n_frames: int = 1,
    ) -> RunHandle:
        """Detect on each video part with the given model + prompt (Section 7).
        Thin wrapper over ``run``."""
        return self.run(
            "detection",
            {
                "model_id": model_id,
                "model_variant": model_variant,
                "prompt": prompt,
                "compute": compute,
                "conf_threshold": conf_threshold,
                "batch_size": batch_size,
                "every_n_frames": every_n_frames,
            },
        )

    def run_tracking(
        self,
        detection_run_id: str,
        tracker_id: str | None = None,
        hyperparams: dict | None = None,
        arena_ids: list[str] | None = None,
        arena_name: str | None = None,
        every_n_frames: int | None = None,
    ) -> RunHandle:
        """Link detections into tracks, optionally arena-filtered (Section 13).
        Thin wrapper over ``run``. ``every_n_frames`` omitted inherits the detection run's.

        ``tracker_id`` omitted takes ``configs.tracking.DEFAULT_TRACKER``, and ``hyperparams``
        omitted takes the chosen tracker's own defaults. Both are left *out* of the config
        rather than passed as ``None``: ``run`` does no None-stripping, and a ``None`` is
        rejected by the model before ``validate`` can resolve it.

        ``arena_name`` is the field of view **by name**, which is the same field of view on
        every recording; ``arena_ids`` names this recording's own arena directly. Prefer the
        name for anything that runs over more than one recording — an id is minted per
        recording, so the same one is a different polygon on the recording next to it.
        """
        config = {
            "detection_run_id": detection_run_id,
            "arena_ids": arena_ids,
            "arena_name": arena_name,
            "every_n_frames": every_n_frames,
        }
        if tracker_id is not None:
            config["tracker_id"] = tracker_id
        if hyperparams is not None:
            config["hyperparams"] = hyperparams
        return self.run("tracking", config)

    # -- keypoints (§26): PR 4 -- the detect stage ------------------------------- #
    def run_keypoint_detect(
        self,
        detection: str,
        model_id: str,
        *,
        example_set_id: str | None = None,
        min_pose_confidence: float | None = None,
        model_variant: str | None = None,
        decode: str | None = None,
        side_rule: str | None = None,
        crop_px: int | None = None,
        every_n_frames: int | None = None,
        batch_size: int | None = None,
        compute: ComputeTarget | None = None,
    ) -> RunHandle:
        """Place a skeleton's points on the animals of a detection run (§26
        ``keypoint_detect``): one row per pose instance, no subject — ``keypoint_track``
        gives each to a tracked subject.

        ``detection`` is the detection run id. ``model_id`` names the method
        (``dinov3_match``, which learns from ``example_set_id``'s marked frames, or a
        distilled student, which keeps every pose at or above ``min_pose_confidence``).
        ``decode`` / ``side_rule`` / ``crop_px`` left unset are resolved against the
        skeleton and written back to the run record; ``every_n_frames`` omitted inherits the
        detection run's cadence. Thin wrapper over ``run``.
        """
        from detail.api import models as api_models

        config = {
            "detection_run_id": detection,
            "model_id": model_id,
            "model_variant": model_variant,
            "example_set_id": example_set_id,
            "min_pose_confidence": min_pose_confidence,
            "decode": decode,
            "side_rule": side_rule,
            "crop_px": crop_px,
            "every_n_frames": every_n_frames,
            "compute": compute
            or default_compute(api_models.supported_backends(model_id)),
        }
        if batch_size is not None:
            config["batch_size"] = int(batch_size)
        return self.run("keypoint_detect", config)

    def keypoint_detect_run(self, run_id: str):
        """A detect run's reader (§26): its record, skeleton, parts, instances, coverage."""
        from detail.storage.keypoint_runs import KeypointDetectRun

        return KeypointDetectRun(self.dir, run_id)

    def keypoint_instances(self, run_id: str, part_id: str) -> dict:
        """Every pose instance a detect run placed in one video part, unassigned (§26
        *Analysis and export*): ``{column: array}`` with ``frame_idx``, ``t_ns``, ``bbox``,
        ``confidence``, ``detection_row``, ``points``, ``point_scores``, ``candidates`` /
        ``candidate_scores`` when kept and ``extras/<name>``. Zero rows for a part where
        nothing was placed.

        Raises:
            KeyError: ``part_id`` is not a video part of this recording.
        """
        from detail.storage import keypoint_runs

        run = self.keypoint_detect_run(run_id)
        if part_id in run.part_ids():
            return run.rows(part_id)
        parts = [
            recording_store.part_stem(f)
            for f in recording_store.video_files(
                recording_store.read_recording(self.dir)
            )
        ]
        if part_id not in parts:
            raise KeyError(
                f"no video part {part_id!r} in recording {self.id!r}; parts: {parts}"
            )
        empty = keypoint_runs._instance_columns(
            len(run.meta["skeleton"]["names"]), run.n_candidates
        )
        del empty["row"]
        return empty

    # -- keypoints (§26): PR 5 -- the track stage ---------------------------------- #
    def run_keypoint_track(
        self,
        detect: str,
        method: str | None = None,
        config: dict | None = None,
        *,
        tracking: str | None = None,
        match_min_inside: float | None = None,
        match_mask_grow: float | None = None,
    ) -> RunHandle:
        """Give a detect run's instances to a tracking run's subjects and settle them over
        time (§26 ``keypoint_track``).

        ``detect`` is the keypoint detect run id; ``tracking`` the tracking run whose
        subjects the instances are assigned to — omitted, the one complete tracking run on
        the detect run's detection run (several are refused, named). ``match_min_inside`` /
        ``match_mask_grow`` left unset take the assignment's defaults. ``method`` omitted
        takes the detect run's method's own default (``dinov3_match``: ``temporal_decode``);
        ``config`` holds the method's settings, every field of which is resolved and
        recorded. CPU only, seconds per recording. Thin wrapper over ``run``.
        """
        run_config = {
            "detect_run_id": detect,
            "tracking_run_id": tracking,
            "method": method,
            "config": dict(config or {}),
        }
        if match_min_inside is not None:
            run_config["match_min_inside"] = float(match_min_inside)
        if match_mask_grow is not None:
            run_config["match_mask_grow"] = float(match_mask_grow)
        return self.run("keypoint_track", run_config)

    def keypoint_track_run(self, run_id: str):
        """A track run's reader (§26): record, skeleton, parts, rows, coverage."""
        from detail.storage.keypoint_runs import KeypointTrackRun

        return KeypointTrackRun(self.dir, run_id)

    def keypoints(self, run_id: str, track_id: int) -> dict[str, dict]:
        """The final points of one track, every part — ``{part_id: {frame_idx, t_ns, points,
        point_scores, changed}}`` (§26 *Surfaces*)."""
        run = self.keypoint_track_run(run_id)
        return {
            part_id: run.rows(track_id, part_id) for part_id in run.part_ids(track_id)
        }

    @property
    def keypoint_detect_runs(self) -> list[dict]:
        """The keypoint detect run records (§26), as ``detection_runs`` is. `run_spec` reads a
        module's runs off the live Recording by ``runs_key``, so without this every picker of
        a keypoint run listed nothing in the app while the dict-driven tests passed."""
        return recording_store.read_recording(self.dir).get("keypoint_detect_runs", [])

    @property
    def keypoint_track_runs(self) -> list[dict]:
        """The keypoint track run records (§26); see ``keypoint_detect_runs``."""
        return recording_store.read_recording(self.dir).get("keypoint_track_runs", [])

    # --- validation (all hard errors, Tenet 10) ----------------------------- #
    # Stage validation lives on each ProcessingModule (§6.5). These two remain
    # for the synchronous inspection tools (§8 prompt-test, §14 render-preview),
    # which are not modules.
    def _validate_render_preview(
        self, data, tracking_run_id, track_ids, renderer_spec, n_sample_frames
    ) -> None:
        if n_sample_frames <= 0:
            raise ValueError(f"n_sample_frames must be > 0, got {n_sample_frames}")
        run = {r["id"]: r for r in data.get("tracking_runs", [])}.get(tracking_run_id)
        if run is None:
            raise ValueError(
                f"tracking run {tracking_run_id!r} not found in recording {self.id!r}"
            )
        if run.get("status") != "complete":
            raise ValueError(
                f"tracking run {tracking_run_id!r} is {run.get('status')!r}, not complete"
            )
        tracks_path = self.dir / "tracking" / tracking_run_id / "tracks.zarr"
        existing = {int(t) for t in PartZarrReader(tracks_path).read()["track_id"]}
        missing = [t for t in track_ids if int(t) not in existing]
        if missing:
            raise ValueError(
                f"track_ids {missing} not in tracking run {tracking_run_id!r}; "
                f"available: {sorted(existing)}"
            )
        unknown = sorted(
            {s["type"] for s in renderer_spec if s["type"] not in TRANSFORMS}
        )
        if unknown:
            raise ValueError(
                f"unknown transform types {unknown}; available: {sorted(TRANSFORMS)}"
            )

    def _validate_prompt_test(
        self,
        data,
        model_id,
        prompts,
        n_sample_frames,
        compute,
        model_variant=None,
        display_classes=None,
    ) -> tuple[bool, list[str]]:
        """Validate a prompt-test request. Returns ``(closed_set, model_classes)`` so
        the caller can build the closed-set class columns. Resolves through the shared
        resolver so a trained/imported model id (not just a pretrained one) is accepted.
        """
        if n_sample_frames <= 0:
            raise ValueError(f"n_sample_frames must be > 0, got {n_sample_frames}")
        model, _ = resolve_detection_model_auto(
            self.dir.parents[1], model_id, model_variant
        )
        if compute.backend not in model.supported_backends:
            raise ValueError(
                f"backend {compute.backend!r} not supported by {model_id!r}; "
                f"supported: {list(model.supported_backends)}"
            )
        closed_set = not model.supported_cue_types
        model_classes = list(getattr(model, "classes", []))
        if closed_set:
            if prompts:
                raise ValueError(
                    f"model {model_id!r} is closed-set and takes no text prompts; "
                    "choose which of its classes to preview instead"
                )
            for name in display_classes or []:
                if name not in model_classes:
                    raise ValueError(
                        f"class {name!r} is not one of {model_id!r}'s classes: "
                        f"{model_classes}"
                    )
        else:
            if display_classes:
                raise ValueError(
                    "display_classes only applies to a closed-set (trained/imported) "
                    f"model; {model_id!r} is open-vocab — pass text prompts instead"
                )
            if not prompts:
                raise ValueError("prompts must be non-empty")
            supported = list(model.supported_cue_types)
            unsupported: set[str] = set()
            for prompt in prompts:
                for target in prompt.targets:
                    if type(target.cue) not in supported:
                        unsupported.add(type(target.cue).__name__)
            if unsupported:
                raise ValueError(
                    f"model {model_id!r} does not support cue types {sorted(unsupported)}; "
                    f"supported: {[c.__name__ for c in supported]}"
                )
        status = model.readiness()
        if not status.ready:
            raise ModelNotReadyError(status)
        files = data.get("video", {}).get("files", [])
        if not files:
            raise ValueError(f"recording {self.id!r} has no video parts")
        for part in files:
            src = recording_store.resolve_part_video_path(self.dir, part)
            if not src.is_file():
                raise ValueError(
                    f"video for part {recording_store.part_stem(part)!r} is missing "
                    f"(expected at {src})"
                )
        return closed_set, model_classes

    # --- recording.yaml bookkeeping ------------------------------------------- #
    def _next_run_id(self, runs: list[dict], prefix: str) -> str:
        nums = []
        for run in runs:
            try:
                nums.append(int(str(run["id"]).rsplit("_", 1)[-1]))
            except (ValueError, KeyError):
                pass
        return f"{prefix}_{max(nums, default=0) + 1:02d}"

    def _reserve_run(self, key: str, prefix: str, make_record) -> str:
        """Allocate the next run id and append its record atomically under the recording lock.

        Serializing the read → allocate → append → write is what lets concurrent runs on one
        recording (e.g. embeddings on separate GPUs) get distinct ids without losing each other's
        records. ``make_record(run_id)`` builds the record once the id is known.
        """
        with recording_store.recording_lock(self.dir):
            data = recording_store.read_recording(self.dir)
            run_id = self._next_run_id(data.get(key, []), prefix)
            data.setdefault(key, []).append(make_record(run_id))
            recording_store.write_recording(self.dir, data)
        return run_id

    def _set_run_status(self, key: str, run_id: str, status: str) -> None:
        with recording_store.recording_lock(self.dir):
            data = recording_store.read_recording(self.dir)
            for run in data.get(key, []):
                if run.get("id") == run_id:
                    run["status"] = status
            recording_store.write_recording(self.dir, data)
