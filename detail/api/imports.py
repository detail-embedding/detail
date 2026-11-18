"""Recording import orchestration (Section 3 — the import reframe).

One import unit is a ``RecordingImportSpec``: an id, grouping variables, and
*optional* ordered video parts + streams. Every entry path (the GUI add-recordings
dialog, the §3 bundle manifest) produces specs; this module executes them against
the experiment API — create the recording + write its metadata row (fast), then import
the video parts in order and attach the streams (the slow, deferrable part) —
setting ``import_status`` so a crashed import is visible, not stuck at ``importing``.

Qt-free, so the GUI import queue and the CLI both drive it. Source paths
(``video``/``raw``/``aux``) are read-only inputs, copied in by the API and never
persisted (Tenet 6).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

from detail.api import recording as recording_api
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.utils import transcode_parallelism
from detail.storage import recording as recording_store
from detail.storage import video as video_store

#: The share of a *copied* part's import spent reading its timestamps; the copy is the rest.
#: Measured on a local disk over parts of different sizes. The low end, because a bar that
#: runs slightly ahead of the scan
#: and then keeps moving through the copy reads better than one that stalls short of it. A
#: referenced part is never copied, so its scan is the whole part (`_phase_spans`).
TIMESTAMP_SHARE = 0.6


@dataclass
class StreamImportSpec:
    name: str
    importer: str
    raw: Path
    aux: dict[str, Path] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    offset_ns: int = 0


@dataclass
class RecordingImportSpec:
    """One recording to import.

    ``name`` is what the user sees — normally the video's filename. ``id`` is the
    recording's permanent identity: left ``None`` it is **minted** by
    :func:`create_shell` (a fresh import, where the tool owns ids), and supplied it is
    **honoured** (restoring a bundle, whose manifest already names its recordings and
    whose other files reference them). Supplying an id and no name means the id *is* the
    name. Once the shell exists, ``id`` is filled in — the later import stages key their
    progress on it.
    """

    name: str = ""
    id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    video_parts: list[Path] = field(default_factory=list)  # ordered; may be empty
    streams: list[StreamImportSpec] = field(default_factory=list)  # may be empty

    def __post_init__(self) -> None:
        if not (self.name or self.id):
            raise ValueError("a recording spec needs a name (or an id to restore)")
        self.name = self.name or self.id

    @property
    def has_deferred_work(self) -> bool:
        return bool(self.video_parts or self.streams)


def build_specs(bundle_dir: Path | str, manifest: dict) -> list[RecordingImportSpec]:
    """Turn a loaded (and validated) bundle manifest into executable specs,
    resolving ``video``/``raw``/``aux`` paths against the bundle folder (§3)."""
    bundle = Path(bundle_dir)
    specs: list[RecordingImportSpec] = []
    for recording in manifest.get("recordings", []):
        streams = [
            StreamImportSpec(
                name=st["name"],
                importer=st["importer"],
                raw=bundle / st["raw"],
                aux={k: bundle / v for k, v in (st.get("aux") or {}).items()},
                config=dict(st.get("config") or {}),
                offset_ns=int(st.get("offset_ns", 0)),
            )
            for st in (recording.get("streams") or [])
        ]
        specs.append(
            RecordingImportSpec(
                # A bundle declares its ids and its own files reference them, so this is a
                # restore: keep them rather than minting new ones.
                id=recording["id"],
                name=recording.get("name") or recording["id"],
                metadata={
                    k: str(v) for k, v in (recording.get("metadata") or {}).items()
                },
                video_parts=[bundle / p for p in (recording.get("video") or [])],
                streams=streams,
            )
        )
    return specs


def create_shell(experiment, spec: RecordingImportSpec):
    """Fast + synchronous: create the recording + seed its metadata row, then set
    ``import_status``. A metadata-only recording is ``ready`` immediately; one with
    video/streams is **``queued``** (a real, tagged recording whose slow video/stream
    import is still pending) until :func:`import_data` runs. Returns the
    ``Recording``."""
    recording = (
        experiment.create_recording(spec.id, metadata=spec.metadata)
        if spec.id
        else experiment.add_recording(spec.name, metadata=spec.metadata)
    )
    spec.id = recording.id  # the later stages key their progress on it
    recording_store.set_import_status(
        recording.dir, "queued" if spec.has_deferred_work else "ready"
    )
    return recording


def create_shells(experiment, specs: list[RecordingImportSpec]) -> None:
    """Create every recording up front (so the whole bout is visible at once):
    metadata-only → ``ready``, video/stream → ``queued``. Fast (filesystem only) —
    safe to call on the UI thread; the slow import is :func:`import_pending`."""
    for spec in specs:
        create_shell(experiment, spec)


def _part_spans(paths: list[Path]) -> list[tuple[float, float]]:
    """Each video part's ``(start, end)`` share of its recording's import, by file size.

    Weighed rather than counted: a recording split at a camera's file-size limit ends in a short
    part, and counting parts would report that sliver as much work as a full one. A file that
    cannot be read weighs nothing; ``add_video_part`` raises the real error for it a moment later.
    """
    sizes = []
    for path in paths:
        try:
            sizes.append(Path(path).stat().st_size)
        except OSError:
            sizes.append(0)
    total = sum(sizes)
    if total:
        weights = [s / total for s in sizes]
    else:
        weights = [1 / len(sizes)] * len(sizes) if sizes else []
    spans, start = [], 0.0
    for weight in weights:
        spans.append((start, start + weight))
        start += weight
    return spans


def _phase_spans(reference: bool) -> dict[str, tuple[float, float]]:
    """Each ``add_video_part`` stage's ``(start, end)`` share of one part."""
    if reference:
        return {recording_api.STAGE_TIMESTAMPS: (0.0, 1.0)}
    return {
        recording_api.STAGE_TIMESTAMPS: (0.0, TIMESTAMP_SHARE),
        recording_api.STAGE_COPYING: (TIMESTAMP_SHARE, 1.0),
    }


class _RecordingProgress:
    """One recording's ``progress_cb(fraction, status)``: throttled, and forced at each phase.

    The fraction is the whole recording's, so it rises through every part and every phase rather
    than restarting at each — a bar that fills on the timestamp scan and then drops to zero for
    the copy is the "99% → 0%" class (DESIGN_PLAN §6, rule 1). ``fraction=None`` holds the last
    one, for a phase with nothing to count.
    """

    def __init__(self, progress_cb: Callable[[float, str], None] | None) -> None:
        sink = (lambda event: progress_cb(*event)) if progress_cb is not None else None
        self._emit = ThrottledProgress(sink)
        self._fraction = 0.0
        self._phase: object = None

    def __call__(self, fraction: float | None, phase: object, status: str) -> None:
        if fraction is not None:
            self._fraction = fraction
        force = phase != self._phase  # the text must never lag the work it names
        self._phase = phase
        self._emit((self._fraction, status), force=force)


def import_data(
    experiment,
    spec: RecordingImportSpec,
    *,
    part_kwargs: dict | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Slow + deferrable: mark the recording ``importing``, import the video parts
    **in order** (Tenet 8 offsets each by the cumulative duration before it), then
    attach the streams, then set the terminal ``import_status``. Assumes the shell
    exists (:func:`create_shell`). ``should_stop`` cancels mid-file → status
    ``cancelled``; other failures → ``failed`` (never stuck at ``importing``)."""
    recording = experiment.get_recording(spec.id)
    part_kwargs = part_kwargs or {}
    n = len(spec.video_parts)
    part_spans = _part_spans(spec.video_parts)
    phases = _phase_spans(part_kwargs.get("storage") == "reference")
    report = _RecordingProgress(progress_cb)
    recording_store.set_import_status(recording.dir, "importing")
    try:
        for i, video in enumerate(spec.video_parts):
            part_start, part_end = part_spans[i]
            where = f" (part {i + 1} of {n})" if n > 1 else ""

            def part_progress(stage, frac, detail, _i=i, _a=part_start, _b=part_end):
                span = phases.get(stage)
                if span is None:  # a phase with no declared share holds the bar
                    fraction = None
                else:
                    lo, hi = span
                    within = lo + (hi - lo) * min(max(frac, 0.0), 1.0)
                    fraction = _a + (_b - _a) * within
                suffix = f" — {detail}" if detail else ""
                report(fraction, (stage, _i), f"{stage}{where}{suffix}")

            recording.add_video_part(
                Path(video),
                progress=part_progress,
                should_stop=should_stop,
                **part_kwargs,
            )
        for j, stream in enumerate(spec.streams):
            # A stream import reports no sub-count, so it holds what the video reached; a
            # recording that is only streams advances one stream at a time.
            report(
                None if n else j / len(spec.streams),
                ("stream", j),
                f"Importing stream {stream.name}",
            )
            recording.add_stream(
                stream.name,
                raw_path=Path(stream.raw),
                importer=stream.importer,
                aux_paths={k: Path(v) for k, v in stream.aux.items()},
                config=stream.config or None,
                offset_ns=stream.offset_ns,
            )
    except (video_store.ImportStopped, KeyboardInterrupt):
        # A Ctrl-C is the user stopping the import, like Cancel. `except Exception` never saw
        # it, so the recording read `importing` until a GUI next opened the experiment.
        recording_store.set_import_status(recording.dir, "cancelled")
        raise
    except Exception:
        recording_store.set_import_status(recording.dir, "failed")
        raise
    # Outside the `try` on purpose: `ready` is a statement *about* a finished import, so the
    # handler that stamps `failed` must not span it. Inside, a transient error writing this one
    # status marked a recording whose video and streams had all imported as `failed` — and
    # `recording.yaml` is precisely the contended file, the one the GUI thread reads while the
    # import worker rewrites it.
    recording_store.set_import_status(recording.dir, "ready")


def _import_one(
    experiment_dir, spec, part_kwargs, progress_q, cancel_event, index, total
):
    """Subprocess worker: import one spec's video/streams. Returns
    ``(id, outcome, error)`` — outcome is ``True`` (imported), ``False`` (an error,
    in ``error``), ``"stopped"`` (cancelled mid-file), or ``"skipped"`` (the batch
    was already cancelled when this worker started). Source paths arrive as
    in-memory pickled args and are never written to a experiment file (Tenet 6)."""
    from detail.api.experiment import Experiment

    if cancel_event.is_set():
        return (spec.id, "skipped", None)
    progress_q.put(("start", spec.id))

    def progress_cb(frac, status):
        progress_q.put(("progress", index, total, spec.id, frac, status))

    try:
        import_data(
            Experiment.open(experiment_dir),
            spec,
            part_kwargs=part_kwargs,
            progress_cb=progress_cb,
            should_stop=cancel_event.is_set,
        )
    except video_store.ImportStopped:
        return (spec.id, "stopped", None)
    except Exception as exc:  # noqa: BLE001 — boundary; report to the parent
        return (spec.id, False, f"{type(exc).__name__}: {exc}")
    return (spec.id, True, None)


def import_pending(
    experiment,
    specs: list[RecordingImportSpec],
    *,
    part_kwargs: dict | None = None,
    on_progress: Callable[[int, int, str, float, str], None] | None = None,
    on_item_started: Callable[[str], None] | None = None,
    on_item_done: Callable[[str, bool, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[tuple[str, str]], bool]:
    """Import each already-created spec (shells from :func:`create_shells` exist).
    A failed spec is recorded and the batch **continues**; on cancel, completed
    recordings are kept and the rest marked ``cancelled`` (tags preserved).
    ``on_progress(index, total, id, fraction, status)``; ``on_item_started(id)``;
    ``on_item_done(id, ok, error)``. Returns ``([(recording_id, error)], cancelled)``.

    CPU transcode is parallelized across recordings when
    :func:`~detail.core.utils.transcode_parallelism` is > 1 and more than one
    recording has deferred work; otherwise it runs sequentially (which also keeps the
    strict cancel-before-each-spec semantics)."""
    deferred = [s for s in specs if s.has_deferred_work]  # empty ones already ready
    try:
        return _import_pending(
            experiment,
            deferred,
            part_kwargs,
            on_progress,
            on_item_started,
            on_item_done,
            should_cancel,
        )
    except KeyboardInterrupt:
        # Ctrl-C in the CLI: nothing after this line runs in this process, and only the GUI
        # reconciles imports on open — so the recordings not reached would read `queued`
        # (and the one mid-file `importing`) until someone opened the GUI (issue #28,
        # finding 14). Mark them the way a Cancel does, on the way out.
        for spec in deferred:
            with contextlib.suppress(Exception):
                rdir = experiment.get_recording(spec.id).dir
                status = recording_store.read_recording(rdir).get("import_status")
                if status in ("queued", "importing"):
                    recording_store.set_import_status(rdir, "cancelled")
        raise


def _import_pending(
    experiment,
    deferred,
    part_kwargs,
    on_progress,
    on_item_started,
    on_item_done,
    should_cancel,
) -> tuple[list[tuple[str, str]], bool]:
    k = min(transcode_parallelism(), len(deferred))
    if k <= 1:
        return _import_pending_serial(
            experiment,
            deferred,
            part_kwargs,
            on_progress,
            on_item_started,
            on_item_done,
            should_cancel,
        )
    return _import_pending_parallel(
        experiment,
        deferred,
        part_kwargs,
        on_progress,
        on_item_started,
        on_item_done,
        should_cancel,
        k,
    )


def _import_pending_serial(
    experiment,
    deferred,
    part_kwargs,
    on_progress,
    on_item_started,
    on_item_done,
    should_cancel,
) -> tuple[list[tuple[str, str]], bool]:
    errors: list[tuple[str, str]] = []
    total = len(deferred)
    for index, spec in enumerate(deferred):
        if should_cancel is not None and should_cancel():
            for remaining in deferred[index:]:  # keep the tagged shells; mark terminal
                recording_store.set_import_status(
                    experiment.get_recording(remaining.id).dir, "cancelled"
                )
            return errors, True
        cb = None
        if on_progress is not None:

            def cb(frac, status, _i=index, _id=spec.id):
                on_progress(_i, total, _id, frac, status)

        if on_item_started is not None:
            on_item_started(spec.id)
        try:
            import_data(
                experiment,
                spec,
                part_kwargs=part_kwargs,
                progress_cb=cb,
                should_stop=should_cancel,
            )
        except video_store.ImportStopped:
            # cancelled mid-file: import_data already marked this one cancelled;
            # mark the rest too and stop the batch.
            for remaining in deferred[index + 1 :]:
                recording_store.set_import_status(
                    experiment.get_recording(remaining.id).dir, "cancelled"
                )
            return errors, True
        except Exception as exc:  # noqa: BLE001 — boundary: record + continue
            errors.append((spec.id, f"{type(exc).__name__}: {exc}"))
            if on_item_done is not None:
                on_item_done(spec.id, False, f"{type(exc).__name__}: {exc}")
            continue
        if on_item_done is not None:
            on_item_done(spec.id, True, "")
    return errors, False


def _import_pending_parallel(
    experiment,
    deferred,
    part_kwargs,
    on_progress,
    on_item_started,
    on_item_done,
    should_cancel,
    k,
) -> tuple[list[tuple[str, str]], bool]:
    # Processes, not threads: the transcode is CPU-bound, and this can run inside
    # the GUI's import QThread, so forking (post-thread) is unsafe -> spawn. Source
    # paths travel as in-memory pickled args (Tenet 6). Each recording writes only its
    # own recording.yaml + files, so concurrent recordings never touch a shared file.
    ctx = multiprocessing.get_context("spawn")
    total = len(deferred)
    experiment_dir = str(experiment.dir)
    errors: list[tuple[str, str]] = []
    cancelled = False
    skipped: list[str] = []

    with ctx.Manager() as manager:
        progress_q = manager.Queue()
        cancel_event = manager.Event()
        if should_cancel is not None and should_cancel():
            cancel_event.set()  # pre-set cancel -> every worker skips deterministically

        def _listen():
            while True:  # drains progress + polls external cancel; exits on None
                if should_cancel is not None and should_cancel():
                    cancel_event.set()
                try:
                    msg = progress_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if msg is None:
                    return
                if msg[0] == "start":
                    if on_item_started is not None:
                        on_item_started(msg[1])
                elif on_progress is not None:  # ("progress", index, total, id, f, s)
                    on_progress(msg[1], msg[2], msg[3], msg[4], msg[5])

        listener = threading.Thread(target=_listen, daemon=True)
        listener.start()
        try:
            with ProcessPoolExecutor(max_workers=k, mp_context=ctx) as pool:
                futures = {
                    pool.submit(
                        _import_one,
                        experiment_dir,
                        spec,
                        part_kwargs or {},
                        progress_q,
                        cancel_event,
                        i,
                        total,
                    ): spec
                    for i, spec in enumerate(deferred)
                }
                for future in as_completed(futures):
                    sid, outcome, error = future.result()
                    if outcome is True:
                        if on_item_done is not None:
                            on_item_done(sid, True, "")
                    elif outcome == "stopped":
                        cancelled = True
                    elif outcome == "skipped":
                        cancelled = True
                        skipped.append(sid)
                    else:  # an error string
                        errors.append((sid, error))
                        if on_item_done is not None:
                            on_item_done(sid, False, error)
        finally:
            progress_q.put(None)  # FIFO: listener drains all prior messages, then exits
            listener.join(timeout=2.0)

        for sid in skipped:  # workers that never started because the batch cancelled
            with contextlib.suppress(FileNotFoundError, ValueError):
                recording_store.set_import_status(
                    experiment.get_recording(sid).dir, "cancelled"
                )

    return errors, cancelled


def import_specs(
    experiment,
    specs: list[RecordingImportSpec],
    *,
    part_kwargs: dict | None = None,
    on_progress: Callable[[int, int, str, float, str], None] | None = None,
    on_item_started: Callable[[str], None] | None = None,
    on_item_done: Callable[[str, bool, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[tuple[str, str]], bool]:
    """Create every shell up front, then import — the Qt-free batch core for the
    CLI + tests. (The GUI splits this: shells on the UI thread, import on a worker.)
    Returns ``([(recording_id, error)], cancelled)``."""
    create_shells(experiment, specs)
    return import_pending(
        experiment,
        specs,
        part_kwargs=part_kwargs,
        on_progress=on_progress,
        on_item_started=on_item_started,
        on_item_done=on_item_done,
        should_cancel=should_cancel,
    )


@dataclass(frozen=True)
class ImportProgress:
    """A whole import batch at one moment — what a surface draws its bar from (§3, §6).

    ``in_flight`` is one ``(name, fraction, phase)`` per recording importing right now, in the
    order they started. It holds more than one when transcode parallelism runs several at once,
    and each one's own fraction counts toward the bar.
    """

    recordings_done: int
    recordings_total: int
    in_flight: tuple[tuple[str, float, str], ...]
    failed: int
    elapsed_s: float
    eta_s: float | None

    @property
    def fraction(self) -> float:
        """``(recordings_done + Σ in-flight fractions) / recordings_total`` — never
        ``recordings_done / recordings_total``, which sits frozen through a long recording.
        """
        if not self.recordings_total:
            return 0.0
        inner = sum(fraction for _, fraction, _ in self.in_flight)
        return min((self.recordings_done + inner) / self.recordings_total, 1.0)


class ImportProgressFold:
    """Fold :func:`import_pending`'s per-recording callbacks into one :class:`ImportProgress`.

    Each callback method returns the new snapshot, so a caller can render it directly. The
    batch can grow while it runs (:meth:`add`): the GUI queue keeps one fold across every
    batch submitted before it next goes idle, so "3 of 21" counts what the user asked for
    rather than restarting with each batch.

    It counts completions, not ``import_pending``'s ``index``: under parallelism ``index`` is a
    recording's place in the batch, not how many are done. Not thread-safe: the GUI feeds it on
    the UI thread, and the CLI feeds it under a lock.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._names: dict[str, str] = {}
        self._in_flight: dict[str, tuple[float, str]] = {}
        self._finished: set[str] = set()
        self._total = 0
        self._failed = 0

    def add(self, specs: list[RecordingImportSpec]) -> ImportProgress:
        """Count ``specs`` that have deferred work. Their shells must exist (ids filled in)."""
        for spec in specs:
            if not spec.has_deferred_work:
                continue
            if spec.id is None:
                raise ValueError(
                    f"recording {spec.name!r} has no id yet; create its shell first"
                )
            self._names[spec.id] = spec.name
            self._total += 1
        return self.snapshot()

    def started(self, recording_id: str) -> ImportProgress:
        if recording_id not in self._finished:
            self._in_flight[recording_id] = (0.0, "starting")
        return self.snapshot()

    def progress(
        self, recording_id: str, fraction: float, phase: str
    ) -> ImportProgress:
        # A parallel worker's last progress message can arrive after its result: the listener
        # thread drains them while the main thread already holds the finished future. Counting
        # it would put a finished recording back in flight, past 100% and never cleared.
        if recording_id not in self._finished:
            self._in_flight[recording_id] = (fraction, phase)
        return self.snapshot()

    def finished(self, recording_id: str, ok: bool) -> ImportProgress:
        if recording_id not in self._finished:
            self._finished.add(recording_id)
            self._in_flight.pop(recording_id, None)
            self._failed += 0 if ok else 1
        return self.snapshot()

    def snapshot(self) -> ImportProgress:
        elapsed = self._clock() - self._started
        event = ImportProgress(
            recordings_done=len(self._finished),
            recordings_total=self._total,
            in_flight=tuple(
                (self._names.get(sid, sid), fraction, phase)
                for sid, (fraction, phase) in self._in_flight.items()
            ),
            failed=self._failed,
            elapsed_s=elapsed,
            eta_s=None,
        )
        return replace(event, eta_s=remaining_seconds(event.fraction, elapsed))


def import_label_batch(
    experiment,
    mapping: list[tuple[str, "Path", list[str] | None]],
    *,
    name: str = "behavior",
    importer: str = "behavior_csv",
    on_progress: Callable[[int, int, str, float, str], None] | None = None,
    on_item_done: Callable[[str, bool, str], None] | None = None,
) -> list[tuple[str, str]]:
    """Import a behavior-label file onto each existing recording (Section 3).

    ``mapping`` is ``[(recording_id, file_path, covered_behaviors), ...]``. Each
    recording's file is added as a ``behavior``-role stream via ``Recording.add_stream``
    (with its declared covered vocabulary, if any), resolved against *its own* active
    timestamp map. A file that fails to parse/resolve is recorded and skipped so one
    bad file never sinks the batch. Reports per recording via ``on_progress(index,
    total, recording_id, fraction, status)``. Returns the skipped recordings as
    ``[(recording_id, error), ...]``.
    """
    total = len(mapping)
    skipped: list[tuple[str, str]] = []
    for index, (sid, file_path, covered) in enumerate(mapping):
        if on_progress is not None:
            on_progress(index, total, sid, 0.0, "importing")
        try:
            experiment.get_recording(sid).add_stream(
                name, file_path, importer, covered_behaviors=covered
            )
            ok, err = True, ""
        except (ValueError, FileNotFoundError, OSError) as exc:
            ok, err = False, str(exc)
            skipped.append((sid, err))
        if on_progress is not None:
            on_progress(index, total, sid, 1.0, "done" if ok else "skipped")
        if on_item_done is not None:
            on_item_done(sid, ok, err)
    return skipped
