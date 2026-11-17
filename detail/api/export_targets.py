"""The export catalogue's entries — what each subject can produce (#336).

Separate from `api/exporters.py` (the contract and the writer) so the contract stays free of the
heavy readers these need: a CLI `--help` and a GUI menu build the catalogue without importing
zarr or cv2, which is what `exporters._load` exists to keep true.

Most targets **wrap a function that already exists** — the point of this change is one contract
and one placement, not seventeen new exporters. What is genuinely new is the §19 clip export,
because both clip exporters that existed then served subjects the build hid (a persisted
clusterer, a syllable-discovery run, both since removed) while the surfaced clustering path had
no export at all.
"""

from __future__ import annotations

import csv
from pathlib import Path

from detail.api import clip_render
from detail.api import export_estimates as est
from detail.api.exporters import (
    ExportError,
    ExportScope,
    ExportSubject,
    ExportTarget,
    experiment_name,
    register,
)
from detail.core.utils import unique_stems
from detail.storage import behavior_output
from detail.storage import experiment as experiment_store

#: PNG for looking at, SVG for a figure that has to be edited or scaled for a paper. PDF is
#: what `report/figures` defaults to and is dropped here: a reader wants one raster and one
#: vector, and a third file per figure is three more things in the folder.
FIGURE_FORMATS = ("png", "svg")


def _figure_style(options: dict):
    """The figure style an export's ``theme`` option asks for — light, for paper, unless told.

    One reading of the option for every target that draws, so a dark export is dark in every
    figure it writes rather than in the ones whose target remembered to look.
    """
    from detail.report.style import DEFAULT_STYLE

    return DEFAULT_STYLE.with_theme(options.get("theme") or "light")


def _draw(dest, stem: str, make) -> list:
    """Render one figure to ``dest/stem`` in every shipped format.

    Every target that shows a picture writes its picture, rather than pictures being a separate
    thing to tick: exporting an ethogram's rows and not the ethogram is the gap this closes, and
    a user who exported "the ethogram" should not have to know there was a second box.
    """
    from detail.report import figures

    return figures.save_figure(make(), Path(dest) / stem, formats=FIGURE_FORMATS)


# --------------------------------------------------------------------------- #
# Resolving a subject to what is on disk
# --------------------------------------------------------------------------- #
def run_record(subject: ExportSubject) -> dict | None:
    """The run record behind a ``run`` / ``analysis_run`` subject, or ``None``."""
    from detail.storage import analysis_runs as analysis_store
    from detail.storage import runs as runs_store

    try:
        if subject.kind == "analysis_run":
            return analysis_store.read_analysis_yaml(
                analysis_store.analysis_dir(subject.experiment_dir, subject.id)
            )
        if subject.kind == "run":
            return runs_store.read_run(subject.experiment_dir, subject.id)
    except (FileNotFoundError, ValueError):
        return None
    return None


def _analysis_dir(subject: ExportSubject) -> Path:
    from detail.storage import analysis_runs as analysis_store

    return analysis_store.analysis_dir(subject.experiment_dir, subject.id)


def _recording_dir(subject: ExportSubject) -> Path:
    return experiment_store.recording_path(subject.experiment_dir, subject.id)


def recordings_of(subject: ExportSubject) -> list[str]:
    """The recording ids a subject spans — what a recordings scope chooses between.

    A processing run's are the recordings it completed. An analysis run's are the ones its
    instances were found in. Anything else (an experiment, one recording, a comparison) has no
    set of recordings to choose from, which is the empty list.
    """
    if subject.kind == "run":
        record = run_record(subject) or {}
        return sorted(
            rid
            for rid, info in (record.get("per_recording") or {}).items()
            if info.get("status") == "complete"
        )
    if subject.kind == "analysis_run":
        try:
            artifact = behavior_output.read_behavior_output(_analysis_dir(subject))
        except (FileNotFoundError, ValueError):
            return []
        return sorted({row.recording_id for row in artifact.instances})
    return []


def _only(scope: ExportScope | None, options: dict) -> set[str] | None:
    """The recordings an export is narrowed to, or ``None`` for every one of them.

    ``scope.recording_ids`` is the answer. ``options`` is accepted so a target that is given a
    ``recording_id`` option by a caller still narrows (the run-level previews on the branch that
    has them seed one); on its own the scope is the whole of it.
    """
    if scope is not None and scope.recording_ids:
        return set(scope.recording_ids)
    chosen = options.get("recording_id")
    return {str(chosen)} if chosen else None


def unavailable_reason(target: ExportTarget, subject: ExportSubject) -> str | None:
    """Why ``target`` cannot run for ``subject``, or ``None`` when it can.

    A sentence naming the **missing thing**, because it is what a greyed-out row says and the
    reader's next question is "so what do I do". "This recording has no streams yet" can be
    acted on; "unavailable" cannot.
    """
    return _REASONS[(target.subject_kind, target.id)](subject)


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
def _run_bundle(subject, dest, scope, progress, **options):
    """The experiment as one archive, reporting from inside the pack rather than around it.

    It used to emit ``packing 0%`` before `export_experiment` and ``packing 100%`` after, so an
    export run's bar sat at zero for the whole of a 17 GB pack (#916). `export_experiment`
    reports its own phases now, and this hands it the callback.
    """
    from detail.core.utils import safe_path_part
    from detail.storage import export as export_store

    # Named after the experiment, so the archive says what it is once it leaves its folder.
    # `.tar` whether or not it is gzipped, as `experiment export --gzip` names it: import sniffs
    # the compression, and the Import picker filters on `*.tar`.
    out = Path(dest) / f"{safe_path_part(experiment_name(subject))}.tar"
    export_store.export_experiment(
        subject.experiment_dir,
        out,
        include_video=bool(options.get("include_video", False)),
        include_datasets=bool(options.get("include_datasets", False)),
        gzip=bool(options.get("gzip", False)),
        progress=progress,
    )
    return [out]


def _run_metadata(subject, dest, scope, progress, **options):
    from detail.api.experiment import Experiment

    out = Path(dest) / "metadata.csv"
    Experiment(subject.experiment_dir).export_metadata_csv(out)
    if progress is not None:
        progress("writing", 1.0, out.name)
    return [out]


register(
    ExportTarget(
        id="bundle",
        label="Experiment bundle",
        subject_kind="experiment",
        artifact="one .tar of the whole experiment, importable back into DETAIL",
        artifact_is_data=True,
        portable=True,
        run=_run_bundle,
        estimate=est.bundle,
    )
)

register(
    ExportTarget(
        id="metadata",
        label="Recording metadata",
        subject_kind="experiment",
        artifact="metadata.csv — one row per recording, one column per grouping variable",
        artifact_is_data=True,
        run=_run_metadata,
        estimate=est.metadata,
    )
)


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def _run_geometry(subject, dest, scope, progress, **options):
    from detail.storage import spatial as spatial_store

    out = Path(dest) / f"{subject.id}_spatial.csv"
    spatial_store.export_spatial(_recording_dir(subject), out)
    if progress is not None:
        progress("writing", 1.0, out.name)
    return [out]


def _video_parts(subject: ExportSubject) -> list[dict]:
    """Every video part on a recording, in order."""
    from detail.storage import recording as recording_store

    try:
        data = recording_store.read_recording(_recording_dir(subject))
    except (FileNotFoundError, ValueError):
        return []
    return list((data.get("video") or {}).get("files", []))


def _copied_parts(subject: ExportSubject) -> list[dict]:
    """Only the parts DETAIL holds a copy of.

    A **reference-in-place** part stores a ``source`` and lives under a machine-local video
    root — it is the user's own file, in a place they chose, and copying it back out is not
    an export of anything DETAIL made.
    """
    return [p for p in _video_parts(subject) if not p.get("source")]


def _canonical_streams(subject: ExportSubject) -> list[dict]:
    """The recording's streams that have a canonical CSV on disk.

    The canonical form is DETAIL's: the raw file the user imported is still theirs, and the
    conversion — one schema, aligned to `t_ns` — is what did not exist before.
    """
    from detail.storage import streams as streams_store

    try:
        listed = streams_store.list_streams(_recording_dir(subject))
    except (FileNotFoundError, ValueError):
        return []
    out = []
    for manifest in listed:
        name = manifest.get("name")
        path = (
            streams_store.stream_dir(_recording_dir(subject), str(name))
            / streams_store.CANONICAL_DIRNAME
            / streams_store.CANONICAL_FILENAME
        )
        if name and path.is_file():
            out.append(manifest)
    return out


def _stream_file_names(recording_id: str, names: list[str]) -> list[str]:
    """``<recording_id>_<stream>.csv`` per stream, unique in the folder even by case (#29)."""
    stems = unique_stems(names, [str(i + 1) for i in range(len(names))])
    return [f"{recording_id}_{stem}.csv" for stem in stems]


def _recording_stems(sources: list) -> list[str]:
    """Each keypoint source's file stem: its recording's name, made path-safe — and, where two
    names clean to one stem (``Mouse 1 (pre)`` / ``Mouse 1 [pre]``), with the recording id after
    it, so neither recording's files are written over the other's (#29)."""
    return unique_stems(
        [s.recording_name for s in sources], [s.recording_id for s in sources]
    )


def _run_streams(subject, dest, scope, progress, **options):
    """Each stream's canonical CSV, named for the stream, plus what each one is."""
    import shutil

    from detail.storage import streams as streams_store

    manifests = _canonical_streams(subject)
    file_names = _stream_file_names(subject.id, [str(m["name"]) for m in manifests])
    written = []
    rows = []
    for index, manifest in enumerate(manifests):
        name = str(manifest["name"])
        source = (
            streams_store.stream_dir(_recording_dir(subject), name)
            / streams_store.CANONICAL_DIRNAME
            / streams_store.CANONICAL_FILENAME
        )
        out = Path(dest) / file_names[index]
        shutil.copyfile(source, out)
        written.append(out)
        rows.append(
            [
                name,
                streams_store.stream_shape(manifest),
                streams_store.stream_origin(manifest),
                manifest.get("role", ""),
                int(manifest.get("offset_ns", 0) or 0),
                out.name,
            ]
        )
        if progress is not None:
            progress("copying", (index + 1) / len(manifests), out.name)
    index_path = Path(dest) / "streams.csv"
    _write_csv(
        index_path,
        ["stream", "shape", "origin", "role", "offset_ns", "file"],
        rows,
    )
    return [*written, index_path]


def _run_timestamps(subject, dest, scope, progress, **options):
    """The PTS-derived `t_ns` for every frame of every part (Tenet 8).

    The one thing on a recording that cannot be recovered from the video: it is read from the
    container's presentation timestamps at import, and every row DETAIL writes is indexed by it.
    Without it, joining a result back to a frame number means trusting the header's nominal fps.
    """
    from detail.storage import recording as recording_store

    recording_dir = _recording_dir(subject)
    data = recording_store.read_recording(recording_dir)
    parts = _video_parts(subject)
    written = []
    for index in range(len(parts)):
        stamps = recording_store.load_active_timestamps(recording_dir, data, index)
        out = Path(dest) / f"{subject.id}_part{index + 1:03d}_timestamps.csv"
        _write_csv(
            out,
            ["frame_idx", "t_ns"],
            [[i, int(t)] for i, t in enumerate(stamps.values)],
        )
        written.append(out)
        if progress is not None:
            progress("writing", (index + 1) / len(parts), out.name)
    return written


def _run_video(subject, dest, scope, progress, **options):
    """The video parts DETAIL holds a copy of, in play order.

    Reference-in-place parts are skipped: those files are already the user's, in the place they
    put them, and the target says so rather than silently writing a shorter recording.

    Copied through `storage.video.copy_with_progress`, the helper importing a video already
    uses, so the bar moves *inside* a part. A whole-file copy reported only at the end of each
    part, and a one-part recording's bar sat at 0% until the export was done (#916). Weighed by
    bytes, because the parts are the cost and they are rarely the same size.
    """
    from detail.core.progress import ThrottledProgress
    from detail.storage import recording as recording_store
    from detail.storage import video as video_store

    recording_dir = _recording_dir(subject)
    sources = []
    for index, part in enumerate(_copied_parts(subject)):
        source = recording_store.resolve_part_video_path(recording_dir, part)
        if source.is_file():
            out = Path(dest) / f"{subject.id}_part{index + 1:03d}{source.suffix}"
            sources.append((source, out))
    total = sum(source.stat().st_size for source, _ in sources) or 1
    emit = ThrottledProgress(
        None if progress is None else (lambda event: progress(*event))
    )
    copied = 0
    written = []
    for source, out in sources:
        emit(("copying", copied / total, out.name), force=True)

        def on_bytes(done, _part_total, _before=copied, _name=out.name):
            emit(("copying", (_before + done) / total, _name))

        copied += video_store.copy_with_progress(source, out, progress=on_bytes)
        written.append(out)
    emit(("copying", 1.0, "done"), force=True)
    return written


def safe_name(name: str) -> str:
    from detail.core.utils import safe_path_part

    return safe_path_part(name, allow_dot=False)


register(
    ExportTarget(
        id="streams",
        label="Streams",
        subject_kind="recording",
        artifact="one CSV per stream, in DETAIL's aligned form",
        data_files=("streams.csv",),
        run=_run_streams,
        estimate=est.streams,
    )
)

register(
    ExportTarget(
        id="timestamps",
        label="Frame timestamps",
        subject_kind="recording",
        artifact="frame_idx,t_ns for every frame of every video part",
        artifact_is_data=True,
        run=_run_timestamps,
        estimate=est.timestamps,
    )
)

register(
    ExportTarget(
        id="video",
        label="Video",
        subject_kind="recording",
        artifact=(
            "the video parts DETAIL imported a copy of, in play order. A part that was "
            "referenced in place is skipped: that file is already yours, under the video root "
            "it was registered from, and copying it back out exports nothing DETAIL made"
        ),
        artifact_is_data=True,
        run=_run_video,
        estimate=est.video,
    )
)


def _run_clip(subject, dest, scope, progress, **options):
    """One mp4 of the frame range in ``scope``, drawn the way the player was drawing it.

    The span is a **scope** (`ExportScope.frames`) rather than an option because it answers the
    question that class exists for — how much of the subject to write — and `describe()` is what
    the manifest's ``scope`` line and the "nothing to write" sentence both read. A range hidden
    in ``**options`` would leave both calling a ten-second clip "everything".

    Which subject to outline (``track_id``) and which run resolves its masks (``embed_run_id``)
    *are* options: they say what to draw, not how much.
    """
    if scope is None or scope.frames is None:
        return []  # `write_export` records this as "had nothing to write", visibly
    from detail.api import clip_export

    embed_run_id = options.get("embed_run_id") or ""
    if not embed_run_id:
        raise ExportError(
            "a clip needs the embedding run its frames were read through — the run that "
            "resolves the tracking and detection behind the masks; pass --embed-run"
        )
    return clip_export.export_clip(
        subject.experiment_dir,
        dest,
        recording_id=subject.id,
        frames=scope.frames,
        embed_run_id=embed_run_id,
        track_id=int(options.get("track_id") or 0),
        renders=options.get("renders") or (clip_render.OVERLAY,),
        crop=bool(options.get("crop", False)),
        size=int(options.get("size", 640)),
        progress=progress,
    )


register(
    ExportTarget(
        id="clip",
        label="Clip",
        subject_kind="recording",
        artifact=(
            "one mp4 of the frame range you picked, at the rate those frames were recorded "
            "at rather than a nominal 30 — so a clip plays back at the speed the behavior "
            "happened"
        ),
        data_files=("clip.csv",),
        scopes=("frames",),
        renders=clip_render.RENDERS,
        run=_run_clip,
        estimate=est.clip,
    )
)


register(
    ExportTarget(
        id="geometry",
        label="Scene geometry",
        subject_kind="recording",
        artifact=(
            "one CSV of the field of view, the distance calibration, and every ROI and POI"
        ),
        artifact_is_data=True,
        run=_run_geometry,
        estimate=est.geometry,
    )
)


# --------------------------------------------------------------------------- #
# analysis run — the §19 behavior artifact
# --------------------------------------------------------------------------- #
def _has_behavior_output(subject: ExportSubject) -> bool:
    return behavior_output.has_behavior_output(_analysis_dir(subject))


def _narrow(rows: list, scope: ExportScope) -> list:
    """``rows`` narrowed to ``scope`` — its recordings, then a cluster or an explicit selection."""
    if scope.recording_ids:
        rows = [r for r in rows if scope.includes(r.recording_id)]
    if scope.instance_ids:
        wanted = set(scope.instance_ids)
        return [r for r in rows if r.instance_id in wanted]
    if scope.cluster is not None:
        want = str(scope.cluster)
        return [r for r in rows if str(r.source_label) == want]
    return rows


_INSTANCE_COLUMNS = [
    "instance_id", "recording_id", "subject", "source_label", "label", "label_path",
    "t_ns", "end_t_ns", "start_frame", "end_frame", "score",
]  # fmt: skip

#: Where each bout happened, appended to the identity columns when the artifact has positions.
#: Export must link clusters to XY, and a cluster id in one file with the
#: coordinates in another is a join the reader has to know exists.
_POSITION_COLUMNS = ["cx_norm", "cy_norm"]

#: Whether one of the user's own clips overlaps the bout — a query run's instances, which its
#: audit never draws or counts (§19). Present only when the artifact recorded it.
_EXAMPLE_COLUMNS = [behavior_output.EXAMPLE_COLUMN]


def _instance_row(row, features: dict | None, example_ids: set | None = None) -> list:
    """One `instances.csv` row; the position pair only when the artifact carries positions.

    ``features`` is `BehaviorOutput.features`, and its three states are three different rows.
    `None` is "this run predates §14's kinematics" — actionable, fixed by
    ``detail recording backfill-kinematics`` — and gets **no columns at all**, so the header says
    the export could not know rather than implying every bout was unplaceable. A present file with
    a `(None, None)` pair is "we looked and this bout had no finite frame", which is an empty cell
    under a header that exists. Collapsing the two into blanks is the `human_label=None` class
    (`tests/storage/test_behavior_audit.py`): two states sharing one encoded value.
    """
    out = [
        row.instance_id, row.recording_id,
        "" if row.subject is None else row.subject,
        row.source_label, row.label, row.label_path,
        row.t_ns, row.end_t_ns, row.start_frame, row.end_frame,
        "" if row.score is None else row.score,
    ]  # fmt: skip
    if features is not None:
        cx, cy = features.get(row.instance_id) or (None, None)
        out += ["" if cx is None else cx, "" if cy is None else cy]
    if example_ids is not None:
        out.append("true" if row.instance_id in example_ids else "false")
    return out


def _write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _run_instances(subject, dest, scope, progress, **options):
    """Every labeled span, plus the coverage that is the denominator of any rate read off it."""
    artifact = behavior_output.read_behavior_output(_analysis_dir(subject))
    rows = _narrow(list(artifact.instances), scope)
    if progress is not None:
        progress("writing", 0.5, f"{len(rows):,} instances")
    if not rows:
        return []
    out = Path(dest) / "instances.csv"
    features = artifact.features
    examples = artifact.example_ids
    columns = (
        _INSTANCE_COLUMNS
        + (_POSITION_COLUMNS if features is not None else [])
        + (_EXAMPLE_COLUMNS if examples is not None else [])
    )
    _write_csv(out, columns, [_instance_row(r, features, examples) for r in rows])

    coverage_path = Path(dest) / "coverage.csv"
    _write_csv(
        coverage_path,
        [
            "recording_id",
            "subject",
            "start_t_ns",
            "end_t_ns",
            "observed_ns",
            "n_observed",
        ],
        [
            [
                c.recording_id,
                "" if c.subject is None else c.subject,
                c.start_t_ns,
                c.end_t_ns,
                c.observed_ns,
                c.n_observed,
            ]
            for c in artifact.coverage
            if scope.includes(c.recording_id)
        ],
    )
    from detail.report import behavior_figures

    if progress is not None:
        progress("drawing", 0.7, "time by behavior")
    style = _figure_style(options)
    drawn = _draw(
        dest,
        "time_by_behavior",
        lambda: behavior_figures.time_by_behavior(rows, style=style),
    )
    if progress is not None:
        progress("drawing", 0.9, "bout durations")
    drawn += _draw(
        dest,
        "bout_durations",
        lambda: behavior_figures.bout_durations(rows, style=style),
    )
    if progress is not None:
        progress("drawing", 1.0, "done")
    return [out, coverage_path, *drawn]


def _run_ethogram(subject, dest, scope, progress, **options):
    """One `start_frame,end_frame,label` CSV per recording — the schema our own importer reads.

    So a result can be re-ingested as a behavior stream, in another tool or back into DETAIL.
    Rows with no human name carry their producer's own id (`source_label`), because an unnamed
    cluster is still a thing that happened; dropping it would silently shorten the ethogram.
    """
    artifact = behavior_output.read_behavior_output(_analysis_dir(subject))
    rows = _narrow(list(artifact.instances), scope)
    by_recording: dict[str, list[list]] = {}
    for row in rows:
        by_recording.setdefault(row.recording_id, []).append(
            [row.start_frame, row.end_frame, row.label or row.source_label]
        )
    written = []
    for index, (recording_id, spans) in enumerate(sorted(by_recording.items())):
        out = Path(dest) / f"{recording_id}_ethogram.csv"
        _write_csv(out, ["start_frame", "end_frame", "label"], sorted(spans))
        written.append(out)
        if progress is not None:
            progress("writing", 0.8 * (index + 1) / len(by_recording), out.name)

    # The picture as well as the rows. "The export ethogram doesn't actually export with plot,
    # just the data" — and an ethogram is a picture before it is a table.
    from detail.report import behavior_figures

    if progress is not None:
        progress("drawing", 0.9, "ethogram")
    style = _figure_style(options)
    coverage = [c for c in artifact.coverage if scope.includes(c.recording_id)]
    written += _draw(
        dest,
        "ethogram",
        lambda: behavior_figures.ethogram(rows, coverage, style=style),
    )
    if progress is not None:
        progress("drawing", 1.0, "done")
    return written


register(
    ExportTarget(
        id="instances",
        label="Behavior instances",
        subject_kind="analysis_run",
        artifact="instances.csv — every labeled span, with its times and frames",
        data_files=("coverage.csv",),
        figure_files=(
            "time_by_behavior.png",
            "time_by_behavior.svg",
            "bout_durations.png",
            "bout_durations.svg",
        ),
        scopes=("cluster", "selection", "recordings"),
        run=_run_instances,
        estimate=est.instances,
    )
)


register(
    ExportTarget(
        id="ethogram",
        label="Ethogram",
        subject_kind="analysis_run",
        artifact=(
            "the ethogram as a picture (PNG + SVG), plus one start_frame,end_frame,label "
            "CSV per recording"
        ),
        artifact_is_data=True,
        figure_files=("ethogram.png", "ethogram.svg"),
        scopes=("cluster", "selection", "recordings"),
        run=_run_ethogram,
        estimate=est.ethogram,
    )
)


def _frame_label_columns(sources: list[str], names: dict[str, list[str]]) -> dict:
    """``{column: [source_label, ...]}`` — one 0/1 column per behavior.

    A named source contributes to each of its names (a name several sources carry is their
    OR); an unnamed one gets a column under its own id, as the ethogram does, because an unnamed
    cluster still happened. A name that equals another source's bare id would silently merge a
    behavior with an unrelated cluster, so it is refused rather than guessed.
    """
    columns: dict[str, list[str]] = {}
    unnamed = [s for s in sources if not names.get(s)]
    for source in sources:
        for name in names.get(source) or []:
            columns.setdefault(name, []).append(source)
    clash = sorted(set(columns) & set(unnamed))
    if clash:
        raise ExportError(
            f"behavior name(s) {clash} are also the id of an unnamed cluster, so their "
            "columns would merge — rename the behavior or name that cluster first"
        )
    for source in unnamed:
        columns[source] = [source]
    return columns


def _run_frame_labels(subject, dest, scope, progress, **options):
    """One CSV per ``(recording, subject)``, one row per video frame: the §19 frame view as the
    regressor a GLM takes.

    ``observed`` is a **column, not a filter**: a frame the tracker lost is kept, with 0 in every
    behavior column and 0 in ``observed``. Dropping or zero-filling it without the flag would
    encode "not analyzed" as "behavior absent" and move the denominator of every rate computed
    off the table — the ``human_label=None`` class (`tests/storage/test_behavior_audit.py`).
    """
    import numpy as np

    from detail.api.experiment import Experiment
    from detail.storage import frame_view

    run_dir = _analysis_dir(subject)
    run = Experiment(subject.experiment_dir).authored_run(subject.id)
    if run is not None:
        run.bring_frame_views_current(progress=progress)
    artifact = behavior_output.read_behavior_output(run_dir)
    names = behavior_output._names_map(artifact.manifest)
    units = [
        unit
        for unit in frame_view.discover_frame_views(run_dir)
        if scope.includes(unit[0])
    ]
    written = []
    for index, (recording_id, unit_subject) in enumerate(units):
        view = frame_view.read_frame_view(run_dir, recording_id, unit_subject)
        sources = [str(s) for s in view.source_labels]
        if scope.cluster is not None:
            sources = [s for s in sources if s == str(scope.cluster)]
        columns = _frame_label_columns(sources, names)
        t_ns = view.t_ns[:]
        observed = view.observed[:].astype(np.int8)
        active = {s: (view.root[s][:] != 0) for s in sources}
        frame_idx = np.arange(len(t_ns), dtype=np.int64)
        rows = slice(None)
        if scope.frames is not None:
            lo, hi = scope.frames
            rows = slice(max(lo, 0), hi + 1)
        stem = (
            recording_id
            if unit_subject is None
            else f"{recording_id}_subject{unit_subject}"
        )
        out = Path(dest) / f"{stem}_frame_labels.csv"
        header = ["frame_idx", "t_ns", "observed", *columns]
        data = [frame_idx[rows], t_ns[rows], observed[rows]]
        for column_sources in columns.values():
            hit = np.zeros(len(t_ns), dtype=bool)
            for s in column_sources:
                hit |= active[s]
            data.append(hit[rows].astype(np.int8))
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(zip(*(col.tolist() for col in data)))
        written.append(out)
        if progress is not None:
            progress("writing", (index + 1) / len(units), out.name)
    return written


register(
    ExportTarget(
        id="frame_labels",
        label="Per-frame behavior labels",
        subject_kind="analysis_run",
        artifact=(
            "one CSV per recording and subject, one row per video frame: frame_idx, t_ns, "
            "observed, and a 0/1 column per behavior"
        ),
        artifact_is_data=True,
        scopes=("cluster", "frames", "recordings"),
        run=_run_frame_labels,
        estimate=est.frame_labels,
    )
)


def _embed_params(subject: ExportSubject) -> dict:
    """The run's own derived ``params``, which name the embedding run it read in each recording.

    Module-agnostic on purpose: every §19 producer's job carries ``embed_run_id`` (and, since
    runs match upstream on settings rather than id, ``embed_run_ids``), so ``runners.job_params``
    puts them in the record and one lookup serves temporal clustering, Query and saved
    pose-clustering runs alike. Asking each module's own state file instead would be three code
    paths and a fourth to write for the next producer. ``core.run_selection`` reads either shape.

    A run recorded before its module derived ``params`` has none, so it reports no clips rather
    than raising when one is asked for — the same rule as any other target whose inputs are not
    on disk.
    """
    record = run_record(subject) or {}
    params = record.get("params")
    return params if isinstance(params, dict) else {}


def _names_an_embedding(subject: ExportSubject) -> bool:
    params = _embed_params(subject)
    return bool(params.get("embed_run_ids") or params.get("embed_run_id"))


def _run_clips(subject, dest, scope, progress, **options):
    """The bouts as video, in whichever flavours were asked for, plus the rows behind them."""
    from detail.api import behavior_clips
    from detail.core import run_selection

    artifact = behavior_output.read_behavior_output(_analysis_dir(subject))
    rows = _narrow(list(artifact.instances), scope)
    if not rows:
        return []
    return behavior_clips.export_behavior_clips(
        subject.experiment_dir,
        dest,
        instances=rows,
        embed_run_ids=run_selection.runs_by_recording(
            _embed_params(subject), {row.recording_id for row in rows}
        ),
        renders=options.get("renders") or (clip_render.OVERLAY,),
        crop=bool(options.get("crop", False)),
        size=int(options.get("size", 640)),
        max_per_cluster=options.get("max_per_cluster"),
        progress=progress,
    )


register(
    ExportTarget(
        id="clips",
        label="Clips",
        subject_kind="analysis_run",
        artifact=(
            "one mp4 per bout, named <cluster>_<recording>_<start>-<end> so a name-sorted "
            "folder plays each cluster back to back"
        ),
        data_files=("clips.csv",),
        scopes=("cluster", "selection", "recordings"),
        renders=clip_render.RENDERS,
        run=_run_clips,
        estimate=est.clips,
    )
)


# --------------------------------------------------------------------------- #
# comparison (§24)
# --------------------------------------------------------------------------- #
def _comparison_record(subject: ExportSubject) -> dict | None:
    from detail.storage import comparisons as comparisons_store

    try:
        return comparisons_store.read_comparison_yaml(
            comparisons_store.comparison_dir(subject.experiment_dir, subject.id)
        )
    except (FileNotFoundError, ValueError, KeyError):
        return None


def _run_comparison(subject, dest, scope, progress, **options):
    """A comparison's figures and its tables — the picture and the numbers under it."""
    from detail.analysis.comparison_report import export_comparison

    record = _comparison_record(subject)
    if record is None:
        return []
    manifest = export_comparison(
        subject.experiment_dir,
        record,
        Path(dest),
        formats=FIGURE_FORMATS,
        style=_figure_style(options),
        progress=progress,
    )
    return [Path(dest) / name for name in sorted(manifest.get("artifacts") or {})]


register(
    ExportTarget(
        id="report",
        label="Comparison report",
        subject_kind="comparison",
        artifact="one figure per measurement of this comparison",
        data_files=("tables/",),
        figure_files=("figures/",),
        run=_run_comparison,
        estimate=est.report,
    )
)


# --------------------------------------------------------------------------- #
# processing runs — the masks, the trajectory, the feature vectors
# --------------------------------------------------------------------------- #
#: Which `run` target each processing module's runs can produce.
#:
#: One declaration, because the alternative is one ``module_id ==`` test per target and each is
#: complete on the day it is written. The instance that showed it was something missing
#: rather than wrong: detection and tracking runs had no entry here at all, so a user on
#: a tracking run was offered `vectors` (refusing) and `performance`, which is the complaint
#: `docs/export_design.md` §1 opens with. `tests/api/test_export_registry.py` holds this to
#: `PROCESSING_MODULES`, so a module added tomorrow has to say how its runs get out.
_RUN_TARGETS: dict[str, tuple[str, ...]] = {
    "detection": ("masks",),
    "tracking": ("tracks",),
    "embedding": ("vectors",),
    # §26 *Analysis and export*: the five keypoint targets read a track run by default and a
    # detect run for its raw, unassigned poses, so both modules produce all five.
    "keypoint_detect": (
        "keypoints",
        "keypoints_dlc",
        "keypoints_sleap",
        "keypoints_nwb",
        "keypoints_video",
    ),
    "keypoint_track": (
        "keypoints",
        "keypoints_dlc",
        "keypoints_sleap",
        "keypoints_nwb",
        "keypoints_video",
    ),
}

#: The keypoint targets, in catalogue order — what the GUI dialog and `keypoints export` list.
KEYPOINT_TARGETS = _RUN_TARGETS["keypoint_track"]

#: How a refusal names the run a target *does* come from. Separate from `_RUN_TARGETS` because
#: that table is the fact and this is the wording; the guard checks this covers exactly the
#: targets that table names, so a new entry there cannot ship without a sentence.
#:
#: Phrased so the sentence does not repeat the target's own name: the GUI renders a greyed row as
#: ``f"{label} — {reason}"``, and "Feature vectors — feature vectors come from an embedding run"
#: is what naming the noun here produces.
_KEYPOINT_SOURCE = (
    "a keypoint track run, or a keypoint detect run for its raw, unassigned poses"
)
_RUN_SOURCE = {
    "masks": "a detection run",
    "tracks": "a tracking run",
    "vectors": "an embedding run",
    "keypoints": _KEYPOINT_SOURCE,
    "keypoints_dlc": _KEYPOINT_SOURCE,
    "keypoints_sleap": _KEYPOINT_SOURCE,
    "keypoints_nwb": _KEYPOINT_SOURCE,
    "keypoints_video": _KEYPOINT_SOURCE,
}


def _produces(record: dict | None, target_id: str) -> bool:
    return target_id in _RUN_TARGETS.get(str((record or {}).get("module_id") or ""), ())


def _wrong_kind(subject: ExportSubject, target_id: str) -> str | None:
    """Why this run is the wrong kind for ``target_id``, or ``None`` when it is the right one."""
    record = run_record(subject)
    if _produces(record, target_id):
        return None
    source = _RUN_SOURCE[target_id]
    module_id = str((record or {}).get("module_id") or "")
    if not module_id:
        return f"this run's record cannot be read; these come from {source}"
    if module_id not in _RUN_TARGETS:
        # A record outlives the module that wrote it (`docs/removed_capabilities.md`).
        return (
            f"this run was made by {module_id!r}, which this version no longer has; "
            f"these come from {source}"
        )
    return f"this is a {module_id} run; these come from {source}"


def _embedding_parts(
    subject: ExportSubject, only: set[str] | None = None
) -> list[tuple[str, int, Path]]:
    """``(recording_id, track_id, part path)`` for every completed part of an embedding run.

    ``only`` narrows it to those recordings (`_only`); ``None`` is all of them.
    """
    record = run_record(subject)
    if not _produces(record, "vectors"):
        return []
    out: list[tuple[str, int, Path]] = []
    for recording_id, info in (record.get("per_recording") or {}).items():
        if info.get("status") != "complete":
            continue
        if only is not None and recording_id not in only:
            continue
        run_dir = (
            experiment_store.recording_path(subject.experiment_dir, recording_id)
            / "embeddings"
            / str(info.get("run_id"))
        )
        if not run_dir.is_dir():
            continue
        for track_dir in sorted(run_dir.glob("track_*")):
            if not track_dir.is_dir():
                continue
            try:
                track_id = int(track_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for part in sorted(track_dir.glob("part_*.zarr")):
                out.append((recording_id, track_id, part))
    return out


def _run_vectors(subject, dest, scope, progress, **options):
    """The feature vectors as one `.npy`, plus the row index that says what each row is.

    `.npy` rather than CSV because a 768-d vector per frame as text is enormous — and useless
    on its own, which is why the index CSV is not optional: without `(recording, track, frame,
    t_ns)` per row the matrix cannot be joined to anything.
    """
    import numpy as np

    from detail.storage.zarr_io import PartZarrReader

    parts = _embedding_parts(subject, _only(scope, options))
    if not parts:
        return []
    blocks, index_rows = [], []
    for step, (recording_id, track_id, path) in enumerate(parts):
        reader = PartZarrReader(path)
        if not reader.n_rows:
            continue
        vectors = np.asarray(reader.column("embedding")[:])
        frames = np.asarray(reader.column("frame_idx")[:])
        t_ns = np.asarray(reader.column("t_ns")[:])
        blocks.append(vectors)
        index_rows.extend(
            [recording_id, track_id, path.stem, int(f), int(t)]
            for f, t in zip(frames, t_ns)
        )
        if progress is not None:
            progress("reading", (step + 1) / len(parts), f"{recording_id} {path.stem}")
    if not blocks:
        return []
    out = Path(dest) / "embeddings.npy"
    np.save(out, np.concatenate(blocks, axis=0))
    index = Path(dest) / "embeddings_index.csv"
    _write_csv(
        index,
        ["row", "recording_id", "track_id", "part", "frame_idx", "t_ns"],
        [[i, *row] for i, row in enumerate(index_rows)],
    )
    return [out, index]


register(
    ExportTarget(
        id="vectors",
        label="Feature vectors",
        subject_kind="run",
        artifact="embeddings.npy — one row per embedded frame",
        data_files=("embeddings_index.csv",),
        scopes=("recordings",),
        run=_run_vectors,
        estimate=est.vectors,
    )
)


def _tracking_runs(
    subject: ExportSubject, only: set[str] | None = None
) -> list[tuple[str, str]]:
    """``(recording_id, tracking run id)`` for every recording this run completed.

    ``only`` narrows it to those recordings (`_only`); ``None`` is all of them.
    """
    record = run_record(subject)
    if not _produces(record, "tracks"):
        return []
    out: list[tuple[str, str]] = []
    for recording_id, info in (record.get("per_recording") or {}).items():
        if info.get("status") != "complete":
            continue
        if only is not None and recording_id not in only:
            continue
        run_id = str(info.get("run_id") or "")
        tracking_dir = (
            experiment_store.recording_path(subject.experiment_dir, recording_id)
            / "tracking"
            / run_id
        )
        if run_id and (tracking_dir / "track.yaml").is_file():
            out.append((recording_id, run_id))
    return sorted(out)


def _run_tracks(subject, dest, scope, progress, **options):
    """The trajectory: one row per (frame, track), one CSV per recording.

    A **format change, not a computation** (`docs/export_design.md` §8), which is why it is an
    export target and the two relation tables beside it are still an analysis: those need the
    recording's ROIs and POIs, and this needs none of that geometry —
    `tests/analysis/test_locations.py::test_no_rois_or_points_just_tracks` is what pins it.

    Goes through `analysis.locations.compute_location_tables` and
    `storage.location_export._write_csv` — the same compute and the same writer
    `LocationExportModule` uses — so the file this writes is byte-identical to the one the
    analysis run writes for the same tracking run. Two doors, one mechanism: the alternative is
    the comparison-figures class, where a second implementation of "the same" export answers
    differently and both files look right.
    """
    from detail.analysis.locations import compute_location_tables
    from detail.api.recording import Recording
    from detail.core.progress import ThrottledProgress
    from detail.storage import location_export

    runs = _tracking_runs(subject, _only(scope, options))
    written = []
    # The compute reports per detection row — hundreds of thousands of them on a real
    # recording — so the throttle is what keeps the emit off the hot loop. Never hand-rolled:
    # `interval_s=None` reads `PROGRESS_INTERVAL_S` at call time, which is the hook a cadence
    # test monkeypatches to make every emit land.
    emit = ThrottledProgress(
        None if progress is None else (lambda event: progress(*event))
    )
    for index, (recording_id, tracking_run_id) in enumerate(runs):

        def on_rows(done, rows_total, _i=index, _rid=recording_id):
            within = done / rows_total if rows_total else 0.0
            emit(("reading", (_i + within) / len(runs), _rid), force=done >= rows_total)

        on_rows(
            0, 1
        )  # a tick per recording, so a single-recording run moves immediately
        tables = compute_location_tables(
            Recording(subject.experiment_dir, recording_id),
            tracking_run_id,
            on_progress=on_rows,
        )
        out = Path(dest) / f"{recording_id}_tracks.csv"
        location_export._write_csv(out, *tables.tracks)
        written.append(out)
    return written


register(
    ExportTarget(
        id="tracks",
        label="Tracks",
        subject_kind="run",
        artifact=(
            "one CSV per recording — a row per frame per track, with the bounding box and the "
            "mask centroid in normalized, pixel and (where the recording is calibrated) "
            "millimetre units"
        ),
        artifact_is_data=True,
        scopes=("recordings",),
        run=_run_tracks,
        estimate=est.tracks,
    )
)


def _detection_parts(
    subject: ExportSubject, only: set[str] | None = None
) -> list[tuple[str, Path]]:
    """``(recording_id, detection run dir)`` for every recording this detection run completed.

    ``only`` narrows it to those recordings (`_only`); ``None`` is all of them.
    """
    record = run_record(subject)
    if not _produces(record, "masks"):
        return []
    out: list[tuple[str, Path]] = []
    for recording_id, info in (record.get("per_recording") or {}).items():
        if info.get("status") != "complete":
            continue
        if only is not None and recording_id not in only:
            continue
        run_dir = (
            experiment_store.recording_path(subject.experiment_dir, recording_id)
            / "detections"
            / str(info.get("run_id"))
        )
        if run_dir.is_dir() and sorted(run_dir.glob("part_*.zarr")):
            out.append((recording_id, run_dir))
    return sorted(out)


def _coco_for_recording(
    experiment_dir: Path, recording_id: str, run_dir: Path, on_rows=None
) -> dict:
    """One recording's detections as a COCO instance-segmentation document.

    **The RLE is passed through, never re-encoded.** `mask_rle` already *is* the COCO counts
    string — the detection worker wrote it with `pycocotools`' own encoder — so
    `core/utils.decode_mask` is not on this path at all and the masks that come out are the
    bytes that went in (`DESIGN_PLAN.md` §17). A decode-and-re-encode would be lossless too, on
    every input where it happened to be, which is exactly the kind of claim that is cheaper to
    make structurally than to test.

    Coordinates are in the **mask resolution**, not the source video's: masks were rasterized
    there, so boxes scaled to anything else would not sit on them. `info` carries both, so a
    consumer that wants video pixels can scale.

    ``on_rows(done, total)`` is called as detection rows are consumed. A recording is hundreds of
    thousands of rows, so the recording boundary is far too coarse to be the only thing that
    moves a bar — the same reason `compute_location_tables` reports per row.

    The document is assembled in memory before it is written, which is COCO's shape rather than a
    choice: one JSON object with cross-referencing id lists cannot be streamed without writing the
    brackets by hand. A 30-minute recording is a file of a few hundred MB at most, seconds
    to build — the same bound `vectors` already works under, which concatenates a
    whole run's embeddings before saving. A recording several times longer would want the parts
    written out separately rather than this made cleverer.
    """
    from pycocotools import mask as mask_utils

    from detail.api.recording import Recording
    from detail.executor import rundir
    from detail.storage import recording as recording_store
    from detail.storage.zarr_io import PartZarrReader

    height, width = (int(v) for v in rundir.read_envelope(run_dir)["mask_resolution"])
    recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
    offsets = recording_store.part_frame_offsets(recording_dir)

    parts = sorted(run_dir.glob("part_*.zarr"))
    rows_total = sum(PartZarrReader(part).n_rows for part in parts)
    rows_done = 0
    images: dict[int, dict] = {}
    annotations: list[dict] = []
    categories: dict[str, int] = {}
    for part_index, part in enumerate(parts):
        columns = PartZarrReader(part).read()
        offset = offsets[part_index] if part_index < len(offsets) else offsets[-1]
        for row in range(len(columns["frame_idx"])):
            rows_done += 1
            if on_rows is not None:
                on_rows(rows_done, rows_total)
            frame_idx = int(columns["frame_idx"][row]) + offset
            if frame_idx not in images:
                images[frame_idx] = {
                    "id": frame_idx,
                    "width": width,
                    "height": height,
                    # The name a frame would have if it were extracted beside this file, so an
                    # annotations-only export and a bundled one name the same image.
                    "file_name": f"{recording_id}/{part.stem}/{frame_idx:06d}.jpg",
                    "t_ns": int(columns["t_ns"][row]),
                }
            class_name = str(columns["class_name"][row])
            category_id = categories.setdefault(class_name, len(categories) + 1)
            x1, y1, x2, y2 = (float(v) for v in columns["bbox"][row])
            counts = str(columns["mask_rle"][row])
            rle = {"counts": counts.encode("ascii"), "size": [height, width]}
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": frame_idx,
                    "category_id": category_id,
                    "segmentation": {"counts": counts, "size": [height, width]},
                    "bbox": [
                        round(x1 * width, 2),
                        round(y1 * height, 2),
                        round((x2 - x1) * width, 2),
                        round((y2 - y1) * height, 2),
                    ],
                    "area": float(mask_utils.area(rle)) if counts else 0.0,
                    "iscrowd": 0,
                    "score": round(float(columns["confidence"][row]), 4),
                }
            )
    resolution = Recording(experiment_dir, recording_id).video_resolution
    return {
        "info": {
            "recording_id": recording_id,
            "detection_run_id": run_dir.name,
            "mask_resolution": [height, width],
            "video_resolution": list(resolution) if resolution else None,
        },
        "images": [images[key] for key in sorted(images)],
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name} for name, index in sorted(categories.items())
        ],
    }


def _run_masks(subject, dest, scope, progress, **options):
    """The detection run's masks as COCO instance-segmentation JSON, one file per recording."""
    import json

    from detail.core.progress import ThrottledProgress

    parts = _detection_parts(subject, _only(scope, options))
    written = []
    emit = ThrottledProgress(
        None if progress is None else (lambda event: progress(*event))
    )
    for index, (recording_id, run_dir) in enumerate(parts):

        def on_rows(done, total, _i=index, _rid=recording_id):
            within = done / total if total else 0.0
            emit(("reading", (_i + within) / len(parts), _rid), force=done >= total)

        on_rows(0, 1)
        document = _coco_for_recording(
            subject.experiment_dir, recording_id, run_dir, on_rows=on_rows
        )
        out = Path(dest) / f"{recording_id}_masks.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        written.append(out)
    if progress is not None:
        progress("reading", 1.0, "done")
    return written


register(
    ExportTarget(
        id="masks",
        label="Masks",
        subject_kind="run",
        artifact=(
            "one COCO instance-segmentation JSON per recording, whose run-length masks are the "
            "bytes DETAIL stored rather than a re-encoding of them"
        ),
        artifact_is_data=True,
        scopes=("recordings",),
        run=_run_masks,
        estimate=est.masks,
    )
)


# --------------------------------------------------------------------------- #
# keypoint runs — §26 "Analysis and export"
# --------------------------------------------------------------------------- #
def _keypoint_runs(
    subject: ExportSubject, only: set[str] | None = None
) -> list[tuple[str, Path, str]]:
    """``(recording_id, recording dir, kd_NN / kt_NN)`` for every recording this keypoint
    run completed and whose stage record is on disk.

    ``only`` narrows it to those recordings (`_only`); ``None`` is all of them."""
    from detail.storage import keypoint_runs

    record = run_record(subject)
    if not _produces(record, "keypoints"):
        return []
    out: list[tuple[str, Path, str]] = []
    for recording_id, info in (record.get("per_recording") or {}).items():
        if info.get("status") != "complete":
            continue
        if only is not None and recording_id not in only:
            continue
        run_id = str(info.get("run_id") or "")
        recording_dir = experiment_store.recording_path(
            subject.experiment_dir, recording_id
        )
        if run_id.startswith("kt_"):
            marker = keypoint_runs.track_run_dir(recording_dir, run_id) / "track.yaml"
        else:
            marker = keypoint_runs.detect_run_dir(recording_dir, run_id) / "detect.yaml"
        if run_id and marker.is_file():
            out.append((recording_id, recording_dir, run_id))
    return sorted(out)


def _keypoint_sources(
    subject: ExportSubject, options: dict, scope: ExportScope | None = None
) -> list:
    """One `PointSource` per completed recording, read the way the options ask: the cut-off,
    ``source`` (the settled track run or the detect run's own placements) and whether a
    point under the cut-off is blanked — decided once here, so every format agrees."""
    from detail.api import keypoint_export

    source = options.get("source") or "track"
    if source not in keypoint_export.SOURCES:
        raise ExportError(
            f"source must be one of {keypoint_export.SOURCES}, not {source!r}"
        )
    cutoff = options.get("cutoff")
    return [
        keypoint_export.PointSource(
            recording_dir,
            run_id,
            source=source,
            cutoff=None if cutoff is None else float(cutoff),
            blank_below_cutoff=bool(options.get("blank_below_cutoff", False)),
        )
        for _recording_id, recording_dir, run_id in _keypoint_runs(
            subject, _only(scope, options)
        )
    ]


def _part_video_name(source, part_id: str) -> str:
    from detail.storage import recording as recording_store

    stems = source.part_stems()
    if part_id in stems:
        part = recording_store.video_files(source.recording)[stems.index(part_id)]
        return recording_store.resolve_part_video_path(source.recording_dir, part).name
    return part_id


def _keypoint_sidecars(dest: Path, sources: list, *, timestamps: bool) -> list[Path]:
    from detail.api import keypoint_export

    written = [
        keypoint_export.write_coverage(
            Path(dest) / keypoint_export.COVERAGE_NAME, sources
        )
    ]
    if timestamps:
        written.append(
            keypoint_export.write_timestamps(
                Path(dest) / keypoint_export.TIMESTAMPS_NAME, sources
            )
        )
    return written


def _run_keypoints(subject, dest, scope, progress, **options):
    """DETAIL's long table: one CSV per recording, a row per frame × track × keypoint (§26),
    pixel and normalized coordinates, the method's score, ``visible`` at the cut-off and
    ``changed`` (0 as detected · 1 another candidate · 2 moved by a filter)."""
    from detail.api import keypoint_export

    sources = _keypoint_sources(subject, options, scope)
    stems = _recording_stems(sources)
    written = []
    for index, source in enumerate(sources):
        out = Path(dest) / f"{stems[index]}_keypoints.csv"
        keypoint_export.write_long_csv(out, [source])
        written.append(out)
        if progress is not None:
            progress("writing", (index + 1) / len(sources), out.name)
    if sources:
        written += _keypoint_sidecars(dest, sources, timestamps=False)
    return written


def _run_keypoints_dlc(subject, dest, scope, progress, **options):
    """DeepLabCut's multi-animal analysis table, one per video part: the CSV always, the H5
    (pandas ``df_with_missing``) when the ``keypoints_export`` extra is installed."""
    from detail.api import keypoint_export

    sources = _keypoint_sources(subject, options, scope)
    stems = _recording_stems(sources)
    written = []
    for index, source in enumerate(sources):
        stem = stems[index]
        for part_id in source.part_ids():
            written += keypoint_export.write_dlc(
                source,
                part_id,
                Path(dest) / f"{stem}_{part_id}_DLC.csv",
                Path(dest) / f"{stem}_{part_id}_DLC.h5",
            )
        if progress is not None:
            progress("writing", (index + 1) / len(sources), stem)
    if sources:
        written += _keypoint_sidecars(dest, sources, timestamps=True)
    return written


def _run_keypoints_sleap(subject, dest, scope, progress, **options):
    """SLEAP's analysis HDF5, one per video part, in SLEAP's own (transposed) layout."""
    from detail.api import keypoint_export

    sources = _keypoint_sources(subject, options, scope)
    stems = _recording_stems(sources)
    written = []
    for index, source in enumerate(sources):
        stem = stems[index]
        for part_id in source.part_ids():
            written.append(
                keypoint_export.write_sleap(
                    source,
                    part_id,
                    Path(dest) / f"{stem}_{part_id}.analysis.h5",
                    _part_video_name(source, part_id),
                )
            )
        if progress is not None:
            progress("writing", (index + 1) / len(sources), stem)
    if sources:
        written += _keypoint_sidecars(dest, sources, timestamps=True)
    return written


def _run_keypoints_nwb(subject, dest, scope, progress, **options):
    """NWB with ndx-pose, one per video part: a ``PoseEstimation`` per track, a series per
    point, the skeleton in the behavior module."""
    from detail.api import keypoint_export

    sources = _keypoint_sources(subject, options, scope)
    stems = _recording_stems(sources)
    written = []
    for index, source in enumerate(sources):
        stem = stems[index]
        for part_id in source.part_ids():
            written.append(
                keypoint_export.write_nwb(
                    source,
                    part_id,
                    Path(dest) / f"{stem}_{part_id}.nwb",
                    video_name=_part_video_name(source, part_id),
                    experiment_name=subject.experiment_dir.name,
                )
            )
        if progress is not None:
            progress("writing", (index + 1) / len(sources), stem)
    if sources:
        written += _keypoint_sidecars(dest, sources, timestamps=False)
    return written


def _run_keypoints_video(subject, dest, scope, progress, **options):
    """The footage with the points drawn on, one mp4 per video part (#689); the tracking
    overlay is opt-in (``overlay``). The points themselves go beside it as ``keypoints.csv``,
    so the picture is never the only copy."""
    from detail.api import keypoint_export, keypoints_video

    sources = _keypoint_sources(subject, options, scope)
    stems = _recording_stems(sources)
    written = []
    for index, source in enumerate(sources):

        def _folded(stage, fraction, detail, _i=index):
            if progress is not None:
                progress(stage, (_i + fraction) / len(sources), detail)

        written += keypoints_video.export_keypoints_video(
            source,
            dest,
            stems[index],
            overlay=bool(options.get("overlay", False)),
            progress=_folded,
        )
    if sources:
        written.append(
            keypoint_export.write_long_csv(Path(dest) / "keypoints.csv", sources)
        )
        written += _keypoint_sidecars(dest, sources, timestamps=False)
    return written


register(
    ExportTarget(
        id="keypoints",
        label="Keypoints",
        subject_kind="run",
        artifact=(
            "one CSV per recording — a row per frame per track per keypoint, in pixel and "
            "normalized units, with the method's score, whether the point is visible at the "
            "cut-off, and what the track stage changed (from a detect run: an empty track and "
            "the pose's instance number on its frame instead); coverage.csv beside it"
        ),
        artifact_is_data=True,
        scopes=("recordings",),
        run=_run_keypoints,
        estimate=est.keypoints,
    )
)

register(
    ExportTarget(
        id="keypoints_dlc",
        label="Keypoints for DeepLabCut",
        subject_kind="run",
        artifact=(
            "one DeepLabCut CSV per video part (and the same table as H5 when the "
            "keypoints_export extra is installed) — scorer / individuals / bodyparts / "
            "coords over x, y, likelihood, a row per frame; what keypoint-MoSeq, VAME, "
            "B-SOiD, SimBA and movement read"
        ),
        data_files=("timestamps.csv", "coverage.csv"),
        scopes=("recordings",),
        run=_run_keypoints_dlc,
        estimate=est.keypoints_dlc,
    )
)

register(
    ExportTarget(
        id="keypoints_sleap",
        label="Keypoints for SLEAP",
        subject_kind="run",
        artifact=(
            "one SLEAP analysis HDF5 per video part — tracks, node_names, track_names, "
            "point_scores, track_occupancy and the skeleton's edges, in SLEAP's own layout"
        ),
        data_files=("timestamps.csv", "coverage.csv"),
        scopes=("recordings",),
        run=_run_keypoints_sleap,
        estimate=est.keypoints_sleap,
    )
)

register(
    ExportTarget(
        id="keypoints_nwb",
        label="Keypoints as NWB",
        subject_kind="run",
        artifact=(
            "one NWB file per video part — a PoseEstimation per track with a "
            "PoseEstimationSeries per point (ndx-pose), timestamps and the skeleton; for "
            "DANDI and the NWB tools"
        ),
        data_files=("coverage.csv",),
        scopes=("recordings",),
        run=_run_keypoints_nwb,
        estimate=est.keypoints_nwb,
    )
)

register(
    ExportTarget(
        id="keypoints_video",
        label="Keypoints video",
        subject_kind="run",
        artifact=(
            "one mp4 per video part with the points drawn on every frame at the display "
            "cut-off, the tracking overlay opt-in"
        ),
        data_files=("keypoints.csv", "coverage.csv"),
        scopes=("recordings",),
        run=_run_keypoints_video,
        estimate=est.keypoints_video,
    )
)


# --------------------------------------------------------------------------- #
# Availability — asked once, at the enumeration point
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Why a target cannot run — asked once, at the enumeration point
# --------------------------------------------------------------------------- #
def _no_experiment(subject) -> str | None:
    return (
        None
        if (subject.experiment_dir / experiment_store.CONFIG_FILENAME).is_file()
        else "this folder is not an experiment"
    )


def _no_artifact(subject) -> str | None:
    return (
        None
        if _has_behavior_output(subject)
        else "this run has not produced a behavior result yet"
    )


def _no_clips(subject) -> str | None:
    reason = _no_artifact(subject)
    if reason is not None:
        return reason
    if not _names_an_embedding(subject):
        return "this run does not record which embedding it came from"
    return None


def _no_geometry(subject) -> str | None:
    from detail.storage import spatial as spatial_store

    recording_dir = _recording_dir(subject)
    if not recording_dir.is_dir():
        return "this recording is not on disk"
    drawn = any(
        spatial_store.list_defs(recording_dir, kind)
        for kind in spatial_store.ALL_SPATIAL_KINDS
    )
    if not drawn:
        return "nothing has been drawn on this recording yet"
    return None


def _no_streams(subject) -> str | None:
    if not _recording_dir(subject).is_dir():
        return "this recording is not on disk"
    if not _canonical_streams(subject):
        return "this recording has no streams imported yet"
    return None


def _no_timestamps(subject) -> str | None:
    if not _video_parts(subject):
        return "this recording has no video parts yet"
    return None


def _no_clip(subject) -> str | None:
    """Video parts only — deliberately not `_copied_parts` like `_no_video` is.

    A rendered clip is a new file DETAIL makes, not a copy of the user's, so a recording whose
    video was referenced in place can still produce one.
    """
    if not _video_parts(subject):
        return "this recording has no video parts to take a clip from"
    return None


def _no_video(subject) -> str | None:
    parts = _video_parts(subject)
    if not parts:
        return "this recording has no video parts yet"
    if not _copied_parts(subject):
        # Short enough to sit on one row: a reason that wraps makes the dialog 1000px wide and
        # is read as a warning rather than an explanation. The tooltip carries the rest.
        return "this video is referenced in place, so the files are already yours"
    return None


def _no_comparison(subject) -> str | None:
    return (
        None
        if _comparison_record(subject) is not None
        else "this comparison is not on disk"
    )


def _no_vectors(subject) -> str | None:
    reason = _wrong_kind(subject, "vectors")
    if reason is not None:
        return reason
    if not _embedding_parts(subject):
        return "this embedding run has no completed output yet"
    return None


def _no_tracks(subject) -> str | None:
    reason = _wrong_kind(subject, "tracks")
    if reason is not None:
        return reason
    if not _tracking_runs(subject):
        return "this tracking run has not completed a recording yet"
    return None


def _no_masks(subject) -> str | None:
    reason = _wrong_kind(subject, "masks")
    if reason is not None:
        return reason
    if not _detection_parts(subject):
        return "this detection run has not completed a recording yet"
    return None


def _no_keypoints(subject) -> str | None:
    reason = _wrong_kind(subject, "keypoints")
    if reason is not None:
        return reason
    if not _keypoint_runs(subject):
        return "this keypoint run has not completed a recording yet"
    return None


def _needs_extra(*modules: str) -> str | None:
    """The optional writer backends, asked by name at enumeration time (no import)."""
    from detail.core.installed import is_importable

    missing = [m for m in modules if not is_importable(m)]
    if not missing:
        return None
    return (
        f"needs {' and '.join(missing)}: install the keypoints_export extra "
        "(pip install 'detail[keypoints_export]')"
    )


def _no_keypoints_sleap(subject) -> str | None:
    return _no_keypoints(subject) or _needs_extra("h5py")


def _no_keypoints_nwb(subject) -> str | None:
    return _no_keypoints(subject) or _needs_extra("pynwb", "ndx_pose")


# --------------------------------------------------------------------------- #
# performance report
# --------------------------------------------------------------------------- #
def _run_dir_for(subject) -> Path | None:
    """Where the run being exported keeps its files, whichever kind of run it is.

    Five run kinds keep their directories in five stores. Asked once here rather than branched
    on at each use, so a sixth kind is one entry rather than an edit at every site.
    """
    from detail.storage import analysis_runs, datasets
    from detail.storage import distill_runs, models
    from detail.storage import runs as runs_store

    finders = {
        "run": runs_store.run_dir,
        "analysis_run": analysis_runs.analysis_dir,
        "dataset": datasets.dataset_dir,
        "model": models.model_dir,
        "distill": distill_runs.run_dir,
    }
    find = finders.get(subject.kind)
    if find is None or not subject.id:
        return None
    try:
        return Path(find(subject.experiment_dir, subject.id))
    except Exception:  # noqa: BLE001 — a missing store is "no report", not a crash
        return None


def _no_activity(subject) -> str | None:
    """Why a performance report is unavailable — which is only ever "this run has no log".

    Offered for a run that *has* one even if the run failed: a failed run's timings are the most
    interesting ones in the file, since the question is usually where it got to. Only a run from
    before the log existed has nothing, and for that the honest answer names the reason rather
    than leaving a greyed row saying "unavailable".
    """
    from detail.storage import activity

    run_dir = _run_dir_for(subject)
    if run_dir is None:
        return "this run's folder is missing"
    if not activity.activity_path(run_dir).is_file():
        return "this run has no activity log — it finished before runs kept one"
    return None


def _job_dirs(subject, run_dir: Path | None) -> dict[str, Path]:
    """Each unit of the run mapped to the directory its worker ran in.

    The work is not always done where the activity log is written. A processing run's log
    belongs to its coordinator, which only relays progress — each recording ran in its own worker,
    under ``recordings/<id>/``. A distill's log belongs to its coordinator too, and its stages ran
    as a dataset job and a training job. The worker is what records the machine and the peak
    memory (``storage/run_facts.py``), so the report has to look where the worker was.
    """
    from detail.executor.runners import PROCESSING_MODULES
    from detail.storage import datasets, distill_runs, models
    from detail.storage import experiment as experiment_store
    from detail.storage import runs as runs_store

    if run_dir is None:
        return {}
    try:
        if subject.kind == "run":
            record = runs_store.read_run(subject.experiment_dir, subject.id)
            module = PROCESSING_MODULES.get(record.get("module_id"))
            if module is None:
                return {}
            out = {}
            for rec_id, entry in (record.get("per_recording") or {}).items():
                if entry.get("run_id"):
                    rec_dir = experiment_store.recording_path(
                        subject.experiment_dir, rec_id
                    )
                    out[rec_id] = rec_dir / module.output_path(entry["run_id"]).rstrip(
                        "/"
                    )
            return out
        if subject.kind == "distill":
            record = distill_runs.read_run(subject.experiment_dir, subject.id)
            stages = record.get("stages") or {}
            out = {}
            dataset_id = (stages.get("label") or {}).get("dataset_id")
            model_id = (stages.get("train") or {}).get("model_id")
            # Keyed by the unit names the coordinator logs (the progress phase), which are
            # not the record's stage names.
            if dataset_id:
                out["labeling"] = datasets.dataset_dir(
                    subject.experiment_dir, dataset_id
                )
            if model_id:
                out["training"] = models.model_dir(subject.experiment_dir, model_id)
            return out
    except (
        Exception
    ):  # noqa: BLE001 — a missing record means "not recorded", not a crash
        return {}
    return {"": run_dir}


def _ran_on(job_dirs: dict[str, Path]) -> dict | None:
    """The machine the run recorded, preferring a job that loaded torch — a GPU job's snapshot
    names the GPU, a CPU-only job's cannot."""
    from detail.core import hostinfo
    from detail.storage import run_facts

    found = [m for m in (run_facts.read_machine(d) for d in job_dirs.values()) if m]
    for machine in found:
        if machine.get("torch") != hostinfo.NOT_LOADED:
            return machine
    return found[0] if found else None


def _peaks(job_dir: Path | None) -> tuple[float | str, float | str]:
    """``(peak RAM GB, peak GPU GB)`` a job recorded — blank where it measured nothing."""
    from detail.storage import run_facts

    resources = run_facts.read_resources(job_dir) if job_dir is not None else None
    if not resources:
        return "", ""

    def gb(value) -> float | str:
        return round(int(value) / 2**30, 2) if value else ""

    return gb(resources.get("peak_ram_bytes")), gb(
        resources.get("peak_gpu_reserved_bytes")
    )


def _progress_counts_frames(subject) -> bool:
    """Whether the run's activity log counted frames, so ``total / seconds`` is a rate.

    Asked of the module that ran (``progress_counts_frames``), not assumed: a module's bar may
    count a per-mille fraction, table rows, or both passes of a two-pass stage. An unknown
    module, a missing record or a distill answers no — a blank rate, never a made-up one.
    """
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import ANALYSIS_MODULES
    from detail.executor.runners import PROCESSING_MODULES
    from detail.executor.runners import import_runner_modules

    registry = {"run": PROCESSING_MODULES, "analysis_run": ANALYSIS_MODULES}.get(
        subject.kind
    )
    if registry is None:
        return False
    module_id = (run_record(subject) or {}).get("module_id")
    import_runner_modules(PIPELINE_RUNNERS)
    module = registry.get(module_id)
    return bool(module is not None and module.progress_counts_frames)


def _run_performance(subject, dest, scope=None, progress=None, **options) -> list[Path]:
    """A machine + timing report for one run, from the log the run already wrote.

    ``**options`` is the target contract (`ExportTarget.run`: *"each target takes what it
    recognises and ignores the rest"*), and this was the one runner of thirteen without it — so
    ticking **Performance report** beside **Clips** in the same export died on
    ``unexpected keyword argument 'renders'``. One export is one call per target with one shared
    options dict, so a target that cannot ignore another target's knobs breaks the combination
    rather than itself.

    What it is for: a table made automatically at the end of every detection run, saying what
    hardware the run used. A local file with a copy button — **no telemetry**, by design. Sharing it is somebody deciding to.

    **The machine is the one the run recorded**, not the one exporting: a run exported on a
    laptop after it ran on a GPU box used to report the laptop. Only a run from before jobs kept
    that record falls back to this computer, and the heading says so.

    **Frames and fps only where the log counted frames.** The log's ``total`` is whatever the
    runner's bar counted, and most analyses count a fraction scaled to 1000 — so the rate is
    shown only for a module that declares ``progress_counts_frames``, and is blank, never
    invented, for every other run (`_progress_counts_frames`). A distill's units are its
    stages, and its log counts a fraction too.

    `performance.md` is the artifact because it pastes into an issue as a table; the CSVs and the
    YAML beside it are the data behind it, which is the catalogue's rule for a target whose
    artifact is a rendering.
    """
    from detail.core import hostinfo
    from detail.core.progress import format_duration
    from detail.storage import activity
    from detail.storage import state_yaml

    run_dir = _run_dir_for(subject)
    rows = activity.read(run_dir) if run_dir is not None else []
    if progress:
        progress("reading", 0.5, "activity log")

    job_dirs = _job_dirs(subject, run_dir)
    recorded = _ran_on(job_dirs)
    facts = recorded or hostinfo.collect()
    per_unit = activity.timings(rows)
    if not _progress_counts_frames(subject):
        per_unit = [{**u, "frames": 0, "fps": ""} for u in per_unit]
    only_job = next(iter(job_dirs.values())) if len(job_dirs) == 1 else None
    for u in per_unit:
        job_dir = job_dirs.get(u["unit"]) or (None if u["unit"] else only_job)
        u["peak_ram_gb"], u["peak_gpu_gb"] = _peaks(job_dir)
    per_phase = activity.phase_timings(rows)
    started, finished, seconds = activity.run_span(rows)
    frames = sum(int(u["frames"] or 0) for u in per_unit)

    machine_yaml = Path(dest) / "machine.yaml"
    state_yaml.write_state_yaml(machine_yaml, {"schema_version": 1, **facts})

    timing_columns = [
        "unit",
        "started",
        "seconds",
        "frames",
        "fps",
        "peak_ram_gb",
        "peak_gpu_gb",
    ]
    timings_csv = Path(dest) / "timings.csv"
    _write_csv(
        timings_csv,
        timing_columns,
        [[u[c] for c in timing_columns] for u in per_unit],
    )
    phases_csv = Path(dest) / "phases.csv"
    _write_csv(
        phases_csv,
        ["unit", "phase", "seconds"],
        [[p["unit"], p["phase"], p["seconds"]] for p in per_phase],
    )

    machine_heading = (
        "## Machine"
        if recorded
        else "## Machine (this computer — the run did not record the one it ran on)"
    )
    lines = [
        f"# {subject.kind} {subject.id} — performance",
        "",
        f"- started: {started or 'unknown'}",
        f"- finished: {finished or 'unknown'}",
        f"- wall clock: {format_duration(seconds) if seconds else 'unknown'}",
        f"- units: {len(per_unit)}",
        f"- frames: {frames:,}" if frames else "- frames: unknown",
        (
            f"- overall: {round(frames / seconds, 1)} fps"
            if frames and seconds > 0
            else "- overall: unknown"
        ),
        "",
        machine_heading,
        "",
        "| | |",
        "| --- | --- |",
        *hostinfo.as_lines(facts),
        "",
        "## Per unit",
        "",
        "| unit | started | wall clock | frames | fps | peak RAM (GB) | peak GPU (GB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for u in per_unit:
        wall = format_duration(u["seconds"]) if u["seconds"] else "—"
        lines.append(
            f"| {u['unit'] or '(whole run)'} | {u['started'] or '—'} | {wall} | "
            f"{u['frames'] or '—'} | {u['fps'] or '—'} | "
            f"{u['peak_ram_gb'] or '—'} | {u['peak_gpu_gb'] or '—'} |"
        )
    if per_phase:
        lines += [
            "",
            "## Per phase",
            "",
            "| unit | phase | wall clock |",
            "| --- | --- | ---: |",
        ]
        for ph in per_phase:
            wall = format_duration(ph["seconds"]) if ph["seconds"] else "—"
            lines.append(f"| {ph['unit'] or '(whole run)'} | {ph['phase']} | {wall} |")
    if not rows:
        lines += [
            "",
            "_This run has no activity log. Runs from before the log existed have none, and "
            "there is nothing to reconstruct one from — the timings above are blank rather "
            "than guessed._",
        ]

    report = Path(dest) / "performance.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if progress:
        progress("writing", 1.0, "performance.md")
    return [report, timings_csv, phases_csv, machine_yaml]


#: Every run kind, because a summary that covers half the tool is a summary somebody has to
#: remember the shape of — and the clustering timings matter most, because temporal clustering can take hours and
#: that is what people want to know first. A
#: distill is a run kind too: its label and train stages are the slowest thing many users start.
for _kind in ("run", "analysis_run", "distill"):
    register(
        ExportTarget(
            id="performance",
            label="Performance report",
            subject_kind=_kind,
            artifact="performance.md — the machine it ran on, and how long each part took",
            data_files=("timings.csv", "phases.csv", "machine.yaml"),
            run=_run_performance,
            estimate=est.performance,
        )
    )


_REASONS = {
    ("experiment", "bundle"): _no_experiment,
    ("experiment", "metadata"): _no_experiment,
    ("recording", "geometry"): _no_geometry,
    ("recording", "streams"): _no_streams,
    ("recording", "timestamps"): _no_timestamps,
    ("recording", "video"): _no_video,
    ("recording", "clip"): _no_clip,
    ("analysis_run", "instances"): _no_artifact,
    ("analysis_run", "ethogram"): _no_artifact,
    ("analysis_run", "frame_labels"): _no_artifact,
    ("analysis_run", "clips"): _no_clips,
    ("analysis_run", "performance"): _no_activity,
    ("distill", "performance"): _no_activity,
    ("comparison", "report"): _no_comparison,
    ("run", "vectors"): _no_vectors,
    ("run", "tracks"): _no_tracks,
    ("run", "masks"): _no_masks,
    ("run", "keypoints"): _no_keypoints,
    ("run", "keypoints_dlc"): _no_keypoints,
    ("run", "keypoints_sleap"): _no_keypoints_sleap,
    ("run", "keypoints_nwb"): _no_keypoints_nwb,
    ("run", "keypoints_video"): _no_keypoints,
    ("run", "performance"): _no_activity,
}
