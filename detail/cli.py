"""CLI wrappers over the Python API (§15).

Each command is a thin click wrapper around a ``Experiment`` / ``Recording`` method.
The CLI never touches ``recording.yaml``, zarr stores, or job dataclasses
directly — it goes through the same API the GUI and tests use (Tenet 2).

Run via the installed entry point (``detail ...``) or ``python -m detail``.

Commands: experiment / recording lifecycle, stream ingestion, the four pipeline runs
(detection, tracking, embedding), experiment-level processing runs
(``process``), dataset export + custom-model training (``dataset`` / ``train``),
and HF credential / model install. Detection prompt-test is here too
(``detection prompt-test``). Deferred: arena / ROI / POI drawing (CLI is a poor
polygon editor — use the GUI) and render-preview (interactive — use the GUI).
"""

import csv
import errno
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from typing import get_args

import click
import tqdm

from detail import console
from detail import docs_access
from detail import surfaces
from detail.api import clip_montage as _clip_montage
from detail.api.experiment import Experiment
from detail.api import tools as api_tools
from detail.cli_numbers import FINITE_FLOAT
from detail.cli_numbers import finite_float_range
from detail.cli_numbers import parse_finite
from detail.cli_paths import UserPath
from detail.configs.stream_select import BorisSelectionConfig, MedpcSessionConfig
from detail.configs.detection import DetectionConfig
from detail.configs.display import ConfidenceView
from detail.configs.embedding import EmbeddingConfig
from detail.configs.labeling import LabelingConfig
from detail.configs.location_export import LocationExportConfig
from detail.configs.temporal_clustering import TemporalClusteringConfig
from detail.configs.temporal_codebook import TemporalCodebookConfig
from detail.configs.query import QueryConfig
from detail.configs.describe import describe_config
from detail.configs.query_eval import QueryEvalConfig
from detail.configs.quick_segment import QuickSegmentConfig
from detail.configs.tracking import TrackingConfig
from detail.configs.windows import WindowSpec
from detail.configs.training import UltralyticsYOLOConfig
from detail.core import buildinfo
from detail.core.dataclasses import (
    ComputeTarget,
    InstallProgress,
    LabelingClass,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.errors import UserInputError
from detail.core.protocols import ModelNotReadyError
from detail.core.utils import best_supported_backend, seconds_to_t_ns
from detail.storage import metadata as metadata_store
from detail.storage import snapshots as snapshots_store
from detail.storage import streams as streams_store
from detail.storage import tool_runs
from detail.storage.export import ExperimentChangedError
from detail.transforms.constructors import DEFAULT_CONSTRUCTOR
from detail.transforms.constructors import surfaced_constructors

# Option help is single-sourced from each module's config model (one home for the text).
_DET = DetectionConfig.model_fields
_EMB = EmbeddingConfig.model_fields
_LAB = LabelingConfig.model_fields
_LOC = LocationExportConfig.model_fields
_QE = QueryEvalConfig.model_fields
_WIN = WindowSpec.model_fields
_QRY = QueryConfig.model_fields
# The clip montage's defaults (§22), read from the API so `--help`, the window and a script
# open on one montage (`tests/api/test_montage_surface_parity.py`).
_MONTAGE_DEFAULTS = _clip_montage.DEFAULT_LAYERS
_MONTAGE_CLIPS = _clip_montage.DEFAULT_CLIPS
_MONTAGE_SORTS = _clip_montage.SORTS
_MONTAGE_SORT = _clip_montage.DEFAULT_SORT
_TEMPORAL = TemporalClusteringConfig.model_fields
_CODEBOOK = TemporalCodebookConfig.model_fields
_TRK = TrackingConfig.model_fields
_TRAIN = UltralyticsYOLOConfig.model_fields
_MEDPC = MedpcSessionConfig.model_fields
_BORIS = BorisSelectionConfig.model_fields
_DET_DOC = {p["name"]: p for p in describe_config(DetectionConfig)}
_EMB_DOC = {p["name"]: p for p in describe_config(EmbeddingConfig)}
_QE_DOC = {p["name"]: p for p in describe_config(QueryEvalConfig)}
_QRY_DOC = {p["name"]: p for p in describe_config(QueryConfig)}
_QSEG_DOC = {p["name"]: p for p in describe_config(QuickSegmentConfig)}
_VIEW_DOC = {p["name"]: p for p in describe_config(ConfidenceView)}


def _int_range(doc: dict) -> click.IntRange:
    """A ``click.IntRange`` from a config field's own bounds — no hand-written range (the
    model is the single source for a parameter's range, per CLAUDE.md)."""
    return click.IntRange(min=doc.get("minimum"), max=doc.get("maximum"))


def _float_range(doc: dict) -> click.ParamType:
    """``_int_range``'s float twin — bounds from the config model, never hand-written, and
    never ``nan`` or ``inf`` (``cli_numbers``)."""
    return finite_float_range(min=doc.get("minimum"), max=doc.get("maximum"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _compute_target(backend: str, devices: tuple[int, ...] | None) -> ComputeTarget:
    return ComputeTarget(
        backend=backend,
        device_indices=list(devices) if devices else None,
        cloud_provider=None,
        cloud_instance=None,
    )


def _open_experiment(experiment_dir: Path) -> Experiment:
    try:
        experiment = Experiment.open(experiment_dir)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(
            f"could not open experiment at {experiment_dir}: {e}"
        )
    # A one-recording run interrupted by Ctrl-C, a closed terminal or a script that never
    # waited leaves `recording.yaml` behind its `job.json`; every command adopts the real
    # status first, as the GUI does on open (issue #28, finding 5).
    experiment.reconcile_recording_runs()
    return experiment


def _stream_config(
    time_column: str | None, time_kind: str, channels: tuple[str, ...]
) -> dict | None:
    """The CSV importer's config from the mapping flags, or ``None`` when none were given.

    ``None`` rather than an all-default dict, so an importer that declares no config model
    (MED-PC, BORIS) still accepts a bare ``add-stream`` — passing it ``{}`` would be a
    config, and refused.
    """
    if time_column is None and time_kind == "auto" and not channels:
        return None
    parsed = []
    for entry in channels:
        column, _, renamed = entry.partition("=")
        if not column:
            raise click.ClickException(
                f"--channel {entry!r} names no column (expected COLUMN or COLUMN=NAME)"
            )
        parsed.append({"column": column, "name": renamed or column})
    return {"time_column": time_column, "time_kind": time_kind, "channels": parsed}


def _selection_config(importer: str | None, **flags) -> dict | None:
    """The MED-PC / BORIS selector flags as that importer's ``config``, or ``None``.

    ``flags`` are ``medpc_<field>`` / ``boris_<field>`` keyword values; only the ones given
    count. A flag for the other importer is refused rather than ignored, since an ignored
    "--boris-subject mA" on a MED-PC file would import exactly what the user meant to narrow.
    """
    given = {k: v for k, v in flags.items() if v is not None}
    if not given:
        return None
    prefixes = {k.split("_", 1)[0] for k in given}
    if len(prefixes) > 1 or importer not in prefixes:
        names = ", ".join("--" + k.replace("_", "-") for k in sorted(given))
        raise click.ClickException(
            f"{names} only apply to the importer they are named for, and this import uses "
            f"{importer!r}"
        )
    return {k.split("_", 1)[1]: v for k, v in given.items()}


def _option_t_ns(seconds: float, what: str) -> int:
    """Seconds typed on the command line as int64 ``t_ns``, through the one checked
    conversion: an offset, a minimum match length, a split gap."""
    try:
        return seconds_to_t_ns(seconds, what=what)
    except ValueError as e:
        raise click.ClickException(str(e)) from None


def _read_stream_sources(
    path: Path,
) -> tuple[bool, list[tuple[str, Path, int | None]]]:
    """Read a `recording,file[,offset_s]` CSV into ``(by_video, sources)`` — what
    ``apply_stream_preset`` takes.

    The first column is a recording's id or name, unless the header names it as a video
    file (``video``, ``file_name`` … — ``metadata_store.VIDEO_KEY_HEADERS``): then each
    cell is a video file name, matched the way a metadata import matches one (#330).
    File paths resolve relative to the table, so the table plus its files move together.
    An absent or empty offset cell is ``None`` ("leave it at the preset's"), which is not
    the same as ``0`` ("this stream starts with the video") -- and a batch is exactly where
    the difference gets lost if the two look alike.

    Raises:
        ValueError: An empty table, a row without both columns, or an unparseable offset.
    """
    rows: list[tuple[str, Path, int | None]] = []
    by_video = False
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line_no, row in enumerate(csv.reader(handle), start=1):
            cells = [cell.strip() for cell in row]
            if not cells or not cells[0] or cells[0].startswith("#"):
                continue
            if line_no == 1 and metadata_store.is_video_key_header(cells[0]):
                by_video = True
                continue
            if line_no == 1 and metadata_store.is_recording_key_header(cells[0]):
                continue  # an optional header, so the file reads as a table
            if len(cells) < 2 or not cells[1]:
                raise ValueError(
                    f"{path}:{line_no}: expected `recording,file[,offset_s]`, got {row!r}"
                )
            offset_ns: int | None = None
            if len(cells) > 2 and cells[2]:
                try:
                    seconds = float(cells[2])
                except ValueError:
                    raise ValueError(
                        f"{path}:{line_no}: offset {cells[2]!r} is not a number of seconds"
                    ) from None
                offset_ns = seconds_to_t_ns(seconds, what=f"{path}:{line_no}: offset")
            rows.append((cells[0], (path.parent / cells[1]).resolve(), offset_ns))
    if not rows:
        raise ValueError(f"{path} lists no recordings")
    return by_video, rows


def _error_text(exc: BaseException) -> str:
    """An exception's message for an ``Error:`` line.

    ``str(KeyError(msg))`` is ``repr(msg)``: quoted, and in double quotes when the message
    itself holds a single quote, which stripping ``'`` left on the line. A ``KeyError`` with a
    message is read from its argument instead.
    """
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _recording(proj: Experiment, ref: str):
    """A recording by **name or id** — what a user types is the name (§11).

    Every ``--recording`` value goes through here: the id is the tool's (opaque,
    permanent) and the name is the user's, so a CLI that only took ids would ask people
    to look up a serial number for something they called ``m27_day3``.
    """
    try:
        return proj.resolve_recording(ref)
    except (KeyError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(_error_text(e))


def _progress_postfix(data: dict) -> str:
    """Compact per-stage extras for the tqdm bar postfix.

    Frames and timing are shown by the bar itself; this surfaces what it can't —
    multi-part position + active-track count (detection / tracking), the current
    track + (track, part) unit position (embedding), and the recording position for
    a recording-keyed analysis run (so it stays on the bar, not just in the message).
    """
    bits: list[str] = []
    units_total = data.get("units_total")
    if units_total:  # embedding: the bar is frames; the postfix carries units
        if data.get("track_id") is not None:
            bits.append(f"track {data['track_id']}")
        if units_total > 1:
            bits.append(f"unit {data.get('units_done', 0)}/{units_total}")
        return " · ".join(bits)
    parts_total = data.get("parts_total")
    if parts_total and parts_total > 1:
        bits.append(f"part {data.get('parts_done', 0)}/{parts_total}")
    if data.get("n_active_tracks") is not None:
        bits.append(f"active {data['n_active_tracks']}")
    recordings_total = data.get("recordings_total")
    if recordings_total and parts_total is None:  # a §16 analysis run (no parts)
        # An event says what its own units are called, the way `LiveRunPanel.set_units` takes
        # its heading as a parameter: an export's units are *targets*, and a bar reading
        # "recording 1/1" through a two-recording masks export is a wrong answer, not a
        # cosmetic one. Defaults to the recording-keyed word every §16 run means.
        noun = data.get("unit_label") or "recording"
        bits.append(
            f"{noun} {_recording_position(data.get('recordings_done', 0), recordings_total)}"
            f"/{recordings_total}"
        )
    return " · ".join(bits)


def _recording_position(done: int, total: int) -> int:
    """1-based index of the recording being worked on, for a done-count of ``done``.

    ``done`` counts *finished* recordings, so the one in flight is ``done + 1`` — except
    on the terminal event, where every recording is finished and ``done == total`` would
    render the nonsensical "recording 2/1" (seen on a real single-recording run).
    """
    return min(int(done) + 1, int(total))


def _interrupted(run, label: str) -> None:
    """What ``Ctrl-C`` means for this run — cancel it, or walk away from it.

    **The handle decides, not the caller.** `RunHandle.owns_worker` is the fact: a run this
    process launched has a worker in its care, and abandoning it orphans a process in its own
    process group that the terminal's interrupt never reached; a *reattached* run's worker
    belongs to whoever launched it, and letting go is what reattaching is for.

    It used to be a `cancel_on_interrupt` parameter on two of the three progress drivers,
    defaulting to cancel. So it reached the two callers somebody was working on — `process
    attach` and `distill attach` — while the other **five** ``attach`` commands (dataset, train,
    analysis, clusterer, export) all stream through `_stream_progress`, which had no parameter
    at all and cancelled. Each of those five says *"Ctrl-C detaches; it does not cancel the
    run"* in its own ``--help``, and has since it was written: a documented promise that the
    function it forwards to had no way to keep, which is two recorded classes at once — an
    opt-in whose default means "as before", and a rule written down and never implemented.

    Raises:
        click.Abort: When the run was cancelled. Detaching is a normal exit, so it returns.
    """
    if not run.owns_worker:
        click.echo(
            f"\n{label}: detached — run {run.id!r} continues in the background",
            err=True,
        )
        return
    click.echo(f"\n{label}: interrupted — cancelling run {run.id!r}…", err=True)
    run.cancel()
    raise click.Abort()


def _stream_progress(run, label: str) -> None:
    """Drive a tqdm progress bar from a RunHandle.progress() iterator.

    Frame-keyed for every stage (``frames_done`` / ``frames_total``) — embedding
    now reports a real run-wide unique-frame total too, so the bar shows a true
    %/ETA and its (track, part) unit position rides in the postfix. Phase
    messages (e.g. "loading model") print above the bar. Writes to stderr so
    stdout stays clean for the final status line.

    The worker runs in its own process group, so a terminal ``Ctrl-C`` reaches
    only this CLI, not the worker. What that should mean depends on whose worker it
    is, and `_interrupted` asks the handle: a run we launched is cancelled rather
    than orphaned, a reattached one is let go.
    """
    bar = None
    last_message = None
    try:
        for event in run.progress():
            data = vars(event)
            # Frame-keyed for the per-frame stages; a recording-keyed event (e.g. clip query,
            # which has no within-unit frames) advances the bar over its recordings_* instead.
            frame_keyed = bool(data.get("frames_total"))
            total = data.get("frames_total") or data.get("recordings_total")
            done = int(
                (
                    data.get("frames_done")
                    if frame_keyed
                    else data.get("recordings_done")
                )
                or 0
            )
            if bar is None:
                bar = tqdm.tqdm(
                    total=total or None,
                    desc=label,
                    unit="frame" if frame_keyed else "step",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            elif total and bar.total != total:
                # A new unit — the next recording of a multi-recording analysis run —
                # restarts its own frame counter. Only moving `total` left tqdm's rate
                # dividing this unit's count by the *whole run's* elapsed, which printed
                # remaining times like 266:07:21. reset() restarts n and the clock
                # together, so the rate describes the unit the postfix already names.
                # _stream_processing_progress has done this on its recording boundary
                # all along; this driver never learned it.
                bar.reset(total=total)
            message = data.get("message")
            if message and message != last_message:
                bar.write(f"  {label}: {message}", file=sys.stderr)
                last_message = message
            bar.n = done
            postfix = _progress_postfix(data)
            if postfix:
                bar.set_postfix_str(postfix, refresh=False)
            bar.refresh()
        run.wait()
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
            bar = None
        _interrupted(run, label)
    finally:
        if bar is not None:
            bar.close()


def _run_failure_reason(run) -> str:
    """Why a run ended without completing, as its record states it, for an ``Error:`` line.

    The reason is what the user needs; pointing them at ``job.json`` made them open a file to
    read one sentence. Only a run that recorded no reason (stopped, or killed before it could
    write one) falls back to where its log is.
    """
    error = run.error
    if error:
        return f": {str(error).strip()}"
    return "; check job.json in the run directory for details."


def _check_status(run, label: str) -> None:
    if run.status != "complete":
        raise click.ClickException(
            f"{label} run {run.id!r} did not complete (status={run.status!r})"
            + _run_failure_reason(run)
        )
    click.echo(f"{label}: {run.id} (status={run.status})")


def _stream_processing_progress(run, label: str) -> None:
    """Drive a tqdm bar from an :class:`AttachedRun`'s ``progress()`` iterator.

    An experiment-level run fans out over recordings: each ``ProcessingProgress`` carries
    the active recording's inner per-stage ``ProgressEvent`` (``frames_*`` + phase
    ``message``). The bar shows the current recording's frame progress and the
    ``recording i/N`` position rides in the postfix; the bar resets at each recording
    boundary. Ctrl-C cancels the run (graceful-then-force) when the caller launched
    it; when reattaching (`RunHandle.owns_worker` is False) it just detaches, leaving the
    detached coordinator to finish the run in the background.
    """
    bar = None
    last_message = None
    current = None
    try:
        for event in run.progress():
            if bar is None:
                bar = tqdm.tqdm(
                    total=None,
                    desc=label,
                    unit="frame",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            inner = event.inner
            if inner is not None:
                data = vars(inner)
                total = data.get("frames_total")
                done = int(data.get("frames_done", 0) or 0)
                if current != event.current_recording:
                    current = event.current_recording
                    bar.reset(total=total or None)
                elif total and bar.total != total:
                    bar.total = total
                message = data.get("message")
                if message and message != last_message:
                    bar.write(
                        f"  {label} [{event.current_recording}]: {message}",
                        file=sys.stderr,
                    )
                    last_message = message
                bar.n = done
            bar.set_postfix_str(
                f"recording {event.current_recording} "
                f"({_recording_position(event.recordings_done, event.recordings_total)}"
                f"/{event.recordings_total})",
                refresh=False,
            )
            bar.refresh()
        run.wait()
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
            bar = None
        _interrupted(run, label)
    finally:
        if bar is not None:
            bar.close()


def _stream_training_progress(run, label: str) -> None:
    """Drive an epoch-keyed tqdm bar from a training ``run.progress()`` iterator.

    Training emits ``TrainingProgressEvent`` (``epoch`` / ``total_epochs`` /
    ``metrics``) — there is no frame total, so the bar counts epochs and shows the
    latest metrics in the postfix. Ctrl-C cancels the run like ``_stream_progress``.
    """
    bar = None
    last_message = None
    try:
        for event in run.progress():
            total = getattr(event, "total_epochs", None)
            done = int(getattr(event, "epoch", 0) or 0)
            if bar is None:
                bar = tqdm.tqdm(
                    total=total or None,
                    desc=label,
                    unit="epoch",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )
            elif total and bar.total != total:
                bar.total = total
            message = getattr(event, "message", None)
            if message and message != last_message:
                bar.write(f"  {label}: {message}", file=sys.stderr)
                last_message = message
            bar.n = done
            metrics = getattr(event, "metrics", None) or {}
            if metrics:
                bar.set_postfix_str(
                    " ".join(f"{k}={v:.3f}" for k, v in list(metrics.items())[:3]),
                    refresh=False,
                )
            bar.refresh()
        run.wait()
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
            bar = None
        click.echo(f"\n{label}: interrupted — cancelling run {run.id!r}…", err=True)
        run.cancel()
        raise click.Abort()
    finally:
        if bar is not None:
            bar.close()


def _stream_distill_progress(run, label: str) -> None:
    """Drive a permille tqdm bar from a distill run's ``DistillProgress`` iterator.

    A distill folds two stages (label → train) into one bar: ``overall_fraction`` drives a
    0–1000 bar, the advancing phase ``message`` is written above it, and ``eta_s`` + the
    ``phase i/N`` position ride in the postfix. Ctrl-C cancels a launched run
    (graceful-then-force); when reattaching (`RunHandle.owns_worker` is False) it detaches,
    leaving the detached coordinator to finish the run in the background.
    """
    bar = None
    last_message = None
    try:
        for event in run.progress():
            if bar is None:
                bar = tqdm.tqdm(
                    total=1000,
                    desc=label,
                    dynamic_ncols=True,
                    file=sys.stderr,
                    # `n` here is the stage-*weighted* fraction, not a count of work, so a
                    # rate estimate over it is meaningless: tqdm's own {remaining} re-derived
                    # the ~5x under-report PR #171 removed from the GUI, printed beside the
                    # runner's correct eta=. DESIGN_PLAN §6's "don't override bar_format"
                    # assumes `n` counts work; it does not here, and the doc carries the
                    # carve-out.
                    bar_format="{l_bar}{bar}| [{elapsed}]{postfix}",
                )
            if event.message and event.message != last_message:
                bar.write(
                    f"  {label} [{event.phase}]: {event.message}", file=sys.stderr
                )
                last_message = event.message
            bar.n = max(0, min(1000, int(event.overall_fraction * 1000)))
            postfix = f"{event.phase} ({event.stage_index + 1}/{event.stage_count})"
            if event.eta_s is not None:
                postfix += f" eta={event.eta_s:.0f}s"
            bar.set_postfix_str(postfix, refresh=False)
            bar.refresh()
        run.wait()
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
            bar = None
        _interrupted(run, label)
    finally:
        if bar is not None:
            bar.close()


_EXPERIMENT_OPT = click.option(
    "--experiment",
    "experiment_dir",
    required=True,
    type=UserPath(file_okay=False, path_type=Path),
    help="Experiment directory.",
)


def _window_options(fn):
    """§23's window model as CLI options — shared by `comparison run` and `measures export`.

    Written once because two commands take the same five knobs. The first copy shipped without
    `--length` / `--step`, so `--windows sliding` silently used the model's defaults with no way
    to say otherwise; a second hand-written copy is where that happens again.

    **Every value comes from `WindowSpec`.** The kinds, the defaults and the help text were all
    hand-typed here — the thing `configs/` exists to make impossible, and a fourth kind added to
    the model would have been unreachable from the CLI while `--help` went on listing three.
    """
    kinds = list(get_args(_WIN["kind"].annotation))
    options = [
        click.option(
            "--windows",
            type=click.Choice(kinds),
            default=_WIN["kind"].default,
            show_default=True,
            help=_WIN["kind"].description,
        ),
        *(
            click.option(
                f"--{flag}", name, type=kind_type,
                default=_WIN[name].default, show_default=True,
                help=_WIN[name].description,
            )
            for flag, name, kind_type in (
                ("every", "every_s", float),
                ("parts", "parts", int),
                ("length", "length_s", float),
                ("step", "step_s", float),
            )
        ),  # fmt: skip
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


def _window_spec(windows, every_s, parts, length_s, step_s) -> dict:
    """The option values as a `configs.windows.WindowSpec` payload."""
    return {
        "kind": windows,
        "every_s": every_s,
        "parts": parts,
        "length_s": length_s,
        "step_s": step_s,
    }


_RECORDING_OPT = click.option(
    "--recording",
    "recording_id",
    required=True,
    type=str,
    help="Recording name (or id) within the experiment.",
)
_RECORDINGS_OPT = click.option(
    "--recording",
    "recording_ids",
    required=True,
    multiple=True,
    type=str,
    help="Recording name (or id) to include (repeatable; at least one).",
)


def _resolve_recording_refs(proj, refs) -> list[str]:
    """Ids for what the user typed. ``--recording`` advertises "name (or id)" and
    ``Experiment.resolve_recording`` accepts either, but the run APIs take ids — so the
    translation has to happen here or the advertised name silently fails to resolve."""
    resolved = []
    for ref in refs:
        try:
            resolved.append(proj.resolve_recording(ref).id)
        except KeyError as exc:
            raise click.ClickException(_error_text(exc))
    return resolved


def _align_opts(command):
    """§14 egocentric alignment, the same five options on ``embedding run`` and ``process
    embedding``; help read off the config model so the three surfaces say one thing."""
    for option in reversed(
        (
            click.option(
                "--align",
                "align",
                type=click.Choice(["none", "rotate", "mirror"]),
                default="none",
                show_default=True,
                help=_EMB["align"].description,
            ),
            click.option(
                "--keypoint-run",
                "keypoint_run_id",
                default=None,
                type=str,
                help=_EMB["keypoint_run_id"].description,
            ),
            click.option(
                "--heading",
                "heading",
                default=None,
                type=str,
                help=_EMB["heading"].description,
            ),
            click.option(
                "--max-gap-s",
                "max_gap_s",
                default=None,
                type=_float_range(_EMB_DOC["max_gap_s"]),
                help=_EMB["max_gap_s"].description,
            ),
            click.option(
                "--mirror-hold-s",
                "mirror_hold_s",
                default=None,
                type=_float_range(_EMB_DOC["mirror_hold_s"]),
                help=_EMB["mirror_hold_s"].description,
            ),
            click.option(
                "--facing-camera-ratio",
                "facing_camera_ratio",
                default=None,
                type=_float_range(_EMB_DOC["facing_camera_ratio"]),
                help=_EMB["facing_camera_ratio"].description,
            ),
        )
    ):
        command = option(command)
    return command


_ALIGN_OPTS = _align_opts


def _alignment_config(align, heading, max_gap_s, mirror_hold_s, facing_camera_ratio):
    """The alignment fields of an embedding config; a setting left out takes the model's
    default, so the record carries a value rather than a hole."""
    defaults = EmbeddingConfig.model_fields
    return {
        "align": align or "none",
        "heading": heading,
        "max_gap_s": (
            defaults["max_gap_s"].default if max_gap_s is None else float(max_gap_s)
        ),
        "mirror_hold_s": (
            defaults["mirror_hold_s"].default
            if mirror_hold_s is None
            else float(mirror_hold_s)
        ),
        "facing_camera_ratio": (
            defaults["facing_camera_ratio"].default
            if facing_camera_ratio is None
            else float(facing_camera_ratio)
        ),
    }


_BACKEND_OPT = click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    show_default=True,
    help="Compute backend.",
)
_DEVICE_OPT = click.option(
    "--device",
    "devices",
    multiple=True,
    type=int,
    help="Device index (repeatable for multi-GPU; cuda only). Left out: GPU 0 alone.",
)
# The CLI twin of the GUI's confidence slider (gui/confidence.ConfidenceRow). These
# commands run the model wide open and keep every proposal, so this filters what gets
# counted, drawn and written — never what runs. Defined once, so both commands say the
# same thing and neither hand-writes the range.
_MIN_CONF_OPT = click.option(
    "--min-conf",
    "min_conf",
    type=_float_range(_VIEW_DOC["min_conf"]),
    default=_VIEW_DOC["min_conf"]["default"],
    show_default=True,
    help=_VIEW_DOC["min_conf"]["help"],
)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
#: The errors the root group shows as ``Error: <message>`` (exit 1) whatever command raised
#: them. A reader that meets an ordinary input mistake raises ``UserInputError`` with a
#: message naming the thing and what to do; a later user-condition error joins this tuple
#: by name. Never ``ValueError`` or ``Exception``: a bug must keep its traceback.
#: ``ExperimentChangedError``: something wrote to the experiment while it was exported.
USER_ERRORS: tuple[type[BaseException], ...] = (UserInputError, ExperimentChangedError)

#: The ``OSError`` numbers the root group shows as one line: the folder or disk the user
#: chose cannot take the write. Any other ``OSError`` (a missing file DETAIL expected to
#: find, a broken pipe) keeps its traceback, so the code that should have checked is found.
USER_OS_ERRNOS: dict[int, str] = {
    errno.EACCES: "permission denied for {file}. Check that you can write to that folder, "
    "or choose another one",
    errno.EPERM: "permission denied for {file}. Check that you can write to that folder, "
    "or choose another one",
    errno.EROFS: "cannot write {file}: the disk is read-only. Copy the experiment to a "
    "disk you can write to, or choose another folder",
    errno.ENAMETOOLONG: "a file or folder name is too long for this disk: {file}. Use a "
    "shorter name",
    errno.ENOSPC: "no space left on the disk while writing {file}. Free some space, then "
    "try again",
}


def user_os_error_message(exc: OSError) -> str | None:
    """The one-line message for an ``OSError`` in :data:`USER_OS_ERRNOS`, else ``None``."""
    template = USER_OS_ERRNOS.get(exc.errno)
    if template is None:
        return None
    target = exc.filename
    if target is None:
        target = "a file"
    elif str(target).endswith(".tmp"):
        # An atomic write's temporary sibling is not a name the user knows: its folder is.
        target = Path(target).parent
    return template.format(file=target)


class _ConsoleSafeGroup(click.Group):
    """The root group, with the output streams hardened before click writes anything.

    ``Command.main`` is the outermost frame of every invocation — the ``detail`` console
    script, ``python -m detail`` and ``CliRunner`` all enter through it — and it runs
    *before* argument parsing. Hooking the group callback instead would miss every eager
    option that prints and exits, ``--help`` among them.
    """

    def main(self, *args: Any, **kwargs: Any) -> Any:
        console.configure_output_streams()
        return super().main(*args, **kwargs)

    def invoke(self, ctx: click.Context) -> Any:
        """Every subcommand runs inside this call, so an ordinary input mistake that no
        command caught still reaches the user as one ``Error:`` line, not a traceback.

        Only the named :data:`USER_ERRORS`, and an ``OSError`` whose number is in
        :data:`USER_OS_ERRNOS`, are turned into a message. Anything else — a ``KeyError``,
        a ``TypeError``, a plain ``ValueError``, a ``FileNotFoundError`` — is a bug in the
        code that raised it and keeps its traceback.
        """
        try:
            return super().invoke(ctx)
        except USER_ERRORS as exc:
            raise click.ClickException(str(exc)) from exc
        except OSError as exc:
            message = user_os_error_message(exc)
            if message is None:
                raise
            raise click.ClickException(message) from exc


def _print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(buildinfo.build_info().label())
    ctx.exit()


@click.group(cls=_ConsoleSafeGroup)
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_print_version,
    help="Print the running build (version, commit date, commit) and exit.",
)
def main() -> None:
    """DETAIL — segmentation / tracking / embedding pipeline for behavioral video."""


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
@main.group()
def experiment() -> None:
    """Experiment lifecycle and inspection."""


@experiment.command("create")
@click.argument("name")
@click.option(
    "--dir",
    "parent_dir",
    required=True,
    type=UserPath(file_okay=False, path_type=Path),
    help="Parent directory to create the experiment inside.",
)
def experiment_create(name: str, parent_dir: Path) -> None:
    """Create a new experiment directory at <parent_dir>/<name>/."""
    try:
        proj = Experiment.create(name, parent_dir)
    except (FileExistsError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"created experiment: {proj.dir}")


@experiment.command("status")
@_EXPERIMENT_OPT
def experiment_status(experiment_dir: Path) -> None:
    """Show experiment identity + recording list + per-recording run counts."""
    proj = _open_experiment(experiment_dir)
    click.echo(f"experiment: {proj.dir}")
    recording_ids = proj.recording_ids
    click.echo(f"recordings: {len(recording_ids)}")
    for recording_id in recording_ids:
        # One damaged recording.yaml (a hand edit, a sync conflict) costs its own line,
        # never the rest of the listing (`dirscan.read_each` is the same trade).
        try:
            sess = proj.get_recording(recording_id)
        except UserInputError as exc:
            click.echo(f"  {recording_id}  (damaged: {exc})")
            continue
        runs = (
            f"det={len(sess.detection_runs)}, "
            f"track={len(sess.tracking_runs)}, "
            f"emb={len(sess.embedding_runs)}"
        )
        click.echo(f"  {sess.name}  ({runs})")  # the name is the handle (§11)


@experiment.command("import-metadata")
@_EXPERIMENT_OPT
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="CSV with a key column + grouping-variable columns (§11).",
)
@click.option(
    "--key",
    "key_column",
    default=None,
    type=str,
    help="The column that links each row to a recording. Default: recording_id, then "
    "name, then a video file name column ("
    + ", ".join(metadata_store.VIDEO_KEY_HEADERS)
    + "; any case). A column named here that is not recording_id or name is matched "
    "as the video file name: base name, any case, extension optional.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the CSV and print the report; apply nothing.",
)
def experiment_import_metadata(
    experiment_dir: Path, csv_path: Path, key_column: str | None, dry_run: bool
) -> None:
    """Bulk-apply recording metadata from a CSV — merge-only, blank cells skipped,
    a row matching no recording rejected (nothing applied on any error)."""
    proj = _open_experiment(experiment_dir)
    errors, warnings, applied = proj.import_metadata_csv(
        csv_path, dry_run=dry_run, key_column=key_column
    )
    for warning in warnings:
        click.echo(f"warning: {warning}")
    if errors:
        for error in errors:
            click.echo(f"error: {error}", err=True)
        raise click.ClickException(f"{len(errors)} error(s); nothing applied")
    if dry_run:
        click.echo("metadata CSV valid — nothing applied (dry run)")
    else:
        click.echo(f"updated {applied} recording(s)")


@experiment.command("export-metadata")
@_EXPERIMENT_OPT
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Write the grouping variables to this CSV (a fill-in template, §11).",
)
def experiment_export_metadata(experiment_dir: Path, csv_path: Path) -> None:
    """Write the experiment's metadata to a CSV template (recording_id, name, file, then
    every column, one row per recording) — fill the blanks offline, then import-metadata.
    """
    proj = _open_experiment(experiment_dir)
    proj.export_metadata_csv(csv_path)
    click.echo(f"wrote {csv_path}")


@experiment.command("add-column")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
def experiment_add_column(experiment_dir: Path, name: str) -> None:
    """Add an (empty) recording-information column, so it can be filled per recording."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.add_metadata_column(name)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"added column: {name}")


@experiment.command("delete-column")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
def experiment_delete_column(experiment_dir: Path, name: str) -> None:
    """Remove a recording-information column and its value on every recording."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.delete_metadata_column(name)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted column: {name}")


@experiment.command("set-metadata")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--set",
    "assignments",
    required=True,
    multiple=True,
    metavar="KEY=VALUE",
    help="Set one recording-information value (repeatable).",
)
def experiment_set_metadata(
    experiment_dir: Path, recording_id: str, assignments: tuple[str, ...]
) -> None:
    """Set recording-information values on one recording.

    A new key becomes a column on every recording (blank elsewhere), the same as adding
    it in the GUI — the table is experiment-wide.
    """
    proj = _open_experiment(experiment_dir)
    recording = _recording(proj, recording_id)
    values = {}
    for assignment in assignments:
        key, sep, value = assignment.partition("=")
        if not sep or not key.strip():
            raise click.ClickException(f"--set expects KEY=VALUE, got {assignment!r}")
        values[key.strip()] = value
    proj.set_recording_metadata(recording.id, values)
    click.echo(f"{recording.name}: set {', '.join(sorted(values))}")


@experiment.command("export")
@_EXPERIMENT_OPT
@click.option(
    "--out",
    "out_path",
    type=UserPath(dir_okay=False, path_type=Path),
    help="Archive path (default <id>.segproj.tar beside the experiment).",
)
@click.option("--include-video", is_flag=True, help="Include source video (large).")
@click.option(
    "--include-datasets", is_flag=True, help="Include training datasets (large)."
)
@click.option("--gzip", is_flag=True, help="gzip the archive.")
def experiment_export(
    experiment_dir: Path,
    out_path: Path | None,
    include_video: bool,
    include_datasets: bool,
    gzip: bool,
) -> None:
    """Pack an experiment into one portable archive (§15) — slim by default (outputs +
    metadata + models; no source video / datasets). scp it to another machine.

    An earlier archive saved inside the experiment is never packed. A folder that is a
    symlink is not followed; each one left out is named.

    Shows a bar on stderr through scanning, checking, hashing and packing."""
    from detail.storage import export as export_store

    proj = _open_experiment(experiment_dir)
    try:
        out = _with_stage_progress(
            lambda bar: proj.export(
                out_path,
                include_video=include_video,
                include_datasets=include_datasets,
                gzip=gzip,
                progress=bar,
            )
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"exported: {out}")
    contents = export_store.read_manifest(out).get("contents") or {}
    for folder in contents.get("symlinked_folders_left_out") or []:
        click.echo(
            f"left out {folder}: it is a symlinked folder, which is not followed"
        )
    _echo_no_video_packed(experiment_dir, contents)


def _echo_no_video_packed(experiment_dir: Path, contents: dict) -> None:
    """Say so when a bundle asked for video packed none (#29): it used to print only
    ``exported:``, and the manifest's ``video_requested: true`` was the only trace."""
    from detail.api.experiment import no_video_packed_reasons

    if contents.get("video_requested") and not contents.get("video_included"):
        click.echo(f"no video was packed: {no_video_packed_reasons(experiment_dir)}")


@experiment.command("import")
@click.argument("archive", type=UserPath(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--replace",
    is_flag=True,
    help="Replace an existing experiment with the same id. Deletes what the bundle does not "
    "bring back (a slim bundle's video and datasets, export records); the list is printed "
    "first.",
)
@click.option("--name", default=None, type=str, help="Import under a different id.")
def experiment_import(archive: Path, replace: bool, name: str | None) -> None:
    """Import an experiment bundle into the managed workspace (§15).

    With --replace, first prints what the replace deletes and the bundle does not bring back:
    video files (count and size), datasets and export records.

    Shows a bar on stderr through reading, extracting, verifying and placing."""
    from detail.api.experiment import replace_warning
    from detail.storage.export import ExperimentExistsError

    def _losses(losses) -> list[str]:
        return [f"  - {line}" for line in replace_warning(losses)]

    if replace:
        try:
            losses = Experiment.replace_losses(archive, name=name)
        except ValueError as e:
            raise click.ClickException(str(e))
        if losses is not None and not losses.empty:
            click.echo(
                f"replacing experiment {losses.experiment_id!r} deletes what the bundle "
                "does not bring back:"
            )
            for line in _losses(losses):
                click.echo(line)
    try:
        proj = _with_stage_progress(
            lambda bar: Experiment.import_archive(
                archive, replace=replace, name=name, progress=bar
            )
        )
    except ExperimentExistsError as e:
        lines = _losses(e.losses)
        raise click.ClickException(
            f"experiment {e.losses.experiment_id!r} already exists. Pass --replace to "
            "replace it, or --name to import a separate copy."
            + ("\n--replace would also delete:\n" + "\n".join(lines) if lines else "")
        )
    except (ValueError, FileExistsError) as e:
        raise click.ClickException(str(e))
    click.echo(f"imported: {proj.dir}")
    missing = proj.referenced_video_missing()
    if missing:
        names = proj.recording_names()
        shown = ", ".join(names.get(r, r) for r in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        click.echo(
            f"video not found on this machine for {len(missing)} recording(s): "
            f"{shown}{more}. (A file at the same path that is a different video counts as "
            "not found.) Relink one file (`detail recording relink`) and every recording "
            "whose video is in the same folder finds it too."
        )


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
@main.group()
def recording() -> None:
    """Recording lifecycle, video ingestion, and inspection."""


@recording.command("create")
@_EXPERIMENT_OPT
@click.option(
    "--name",
    required=True,
    type=str,
    help="The recording's name: what every command's --recording and the GUI take. "
    "The tool assigns its id (recording_000, …).",
)
def recording_create(experiment_dir: Path, name: str) -> None:
    """Create a new, empty recording named NAME; the tool assigns its id."""
    proj = _open_experiment(experiment_dir)
    name = name.strip()
    if not name:
        raise click.ClickException("--name cannot be empty")
    # A name a user typed is refused when taken, never suffixed the way the importer's
    # filenames are (Recording.set_name, CLAUDE.md "A recording's id is the tool's").
    holder = next(
        (rid for rid, existing in proj.recording_names().items() if existing == name),
        None,
    )
    if holder is not None:
        raise click.ClickException(
            f"another recording is already named {name!r} (id {holder}); "
            "names identify recordings to you, so they have to be unique"
        )
    recording = proj.add_recording(name)
    click.echo(f"created recording: {recording.name}  (id {recording.id})")


class _StageProgress:
    """Drive one tqdm bar from a synchronous ``(stage, fraction, detail)`` callback.

    The import moves through phases ("Reading timestamps" → "Copying"); on a stage
    change the bar is re-labeled and its count reset.
    Percent-based (``total=100``); writes to stderr so stdout stays clean for the
    final status line. Long imports (the copy + timestamp scan of a multi-hour
    video) would otherwise sit silent — the CLI's other long operations all show a
    bar via ``_stream_progress`` (this is the synchronous analogue)."""

    def __init__(self) -> None:
        self._bar = None
        self._stage: str | None = None

    def __call__(self, stage: str, fraction: float, detail: str) -> None:
        if self._bar is None:
            self._bar = tqdm.tqdm(
                total=100, desc=stage, dynamic_ncols=True, file=sys.stderr
            )
            self._stage = stage
        elif stage != self._stage:
            self._bar.set_description(stage)
            self._bar.reset(total=100)
            self._stage = stage
        self._bar.n = min(100, max(0, round(fraction * 100)))
        if detail:
            self._bar.set_postfix_str(detail, refresh=False)
        self._bar.refresh()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _import_with_progress(proj, specs) -> list[tuple[str, str]]:
    """Import ``specs`` under one bar for the whole batch; returns the failures.

    The CLI's mapping for ``api.imports.ImportProgress``: the bar is the batch's folded
    fraction (finished recordings plus each in-flight one's own), and the postfix always says
    recordings done/total plus the phase of what is importing now. One stage label for the
    whole batch, so ``_StageProgress`` never resets it; tqdm's ETA counts recordings.
    """
    from detail.api.imports import ImportProgressFold, create_shells, import_pending

    create_shells(proj, specs)
    fold = ImportProgressFold()
    fold.add(specs)
    bar = _StageProgress()
    # Parallel imports report from a listener thread while results land on this one.
    lock = threading.Lock()

    def render(update) -> None:
        with lock:
            event = update()
            bar("Importing", event.fraction, _import_detail(event))

    try:
        failed, _ = import_pending(
            proj,
            specs,
            on_progress=lambda _i, _n, sid, frac, phase: render(
                lambda: fold.progress(sid, frac, phase)
            ),
            on_item_started=lambda sid: render(lambda: fold.started(sid)),
            on_item_done=lambda sid, ok, _err: render(lambda: fold.finished(sid, ok)),
        )
    finally:
        bar.close()
    return failed


def _import_detail(event) -> str:
    """``3/21 done · mouse_07_day1: Copying — 1.2 / 2.9 GB`` — always the count, then the work."""
    detail = f"{event.recordings_done}/{event.recordings_total} done"
    if event.failed:
        detail += f", {event.failed} failed"
    if event.in_flight:
        name, _, phase = event.in_flight[0]
        others = len(event.in_flight) - 1
        detail += f" · {name}: {phase}" + (f" (+{others} more)" if others else "")
    return detail


def _with_stage_progress(call):
    """Run ``call(bar)`` with a :class:`_StageProgress` bar, closed however it ends.

    For an API call that is only *sometimes* long — a pin that has deferred frame views to
    rebuild — the bar stays unopened when nothing reports, so the quick case prints nothing.
    """
    bar = _StageProgress()
    try:
        return call(bar)
    finally:
        bar.close()


@recording.command("add-video")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--video",
    "video_path",
    required=True,
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Video file to import as the next part.",
)
@click.option(
    "--reference",
    is_flag=True,
    help="Reference the video in place (don't copy it in). The recording stores the "
    "file's location + a content fingerprint; the location is remembered automatically.",
)
@click.option(
    "--video-root",
    "video_root",
    metavar="ALIAS",
    help="Advanced: reference under a specific registered video root (see `video-root "
    "set`) rather than the auto-remembered folder. Implies --reference.",
)
@click.option(
    "--no-preview",
    "no_preview",
    is_flag=True,
    help="Skip building the display-only preview after the import (build it later with "
    "`cache previews build`).",
)
def recording_add_video(
    experiment_dir: Path,
    recording_id: str,
    video_path: Path,
    reference: bool,
    video_root: str | None,
    no_preview: bool,
) -> None:
    """Import a video file as the next part of a recording.

    The video is copied in at its original rate and resolution; a stage that wants
    fewer frames sets ``--every-n-frames`` on the run instead. With ``--reference`` the
    video is referenced in place (not copied); its folder is remembered automatically
    (or pass ``--video-root ALIAS`` to pin a specific one).

    Afterwards the part's display-only preview is built into this machine's preview
    cache, with its own bar; ``--no-preview`` skips it.
    """
    proj = _open_experiment(experiment_dir)
    bar = _StageProgress()
    referenced = reference or video_root is not None
    try:
        sess = _recording(proj, recording_id)
        sess.add_video_part(
            video_path,
            storage="reference" if referenced else "default",
            video_root=video_root,
            progress=bar,
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    how = "referenced" if referenced else "copied"
    click.echo(f"added video to {recording_id} ({how}): {video_path.name}")
    if not no_preview:
        _build_preview_after_import(proj, sess, experiment_dir)


def _build_preview_after_import(proj: Experiment, sess, experiment_dir: Path) -> None:
    """Build a just-imported recording's preview, and **warn** — never fail — when it does not.

    The import is what the command was for, and it succeeded: a preview is a display-only
    convenience (§3) that ``cache previews build`` rebuilds at any time, so a non-zero exit
    here told a script the video was not added when it was.
    """
    bar = _StageProgress()
    report, error = None, ""
    try:
        report = sess.build_preview(progress=bar, masks=False)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        error = str(e)
    finally:
        bar.close()
    if report is not None and report.ok:
        return
    if report is None:
        click.echo(f"preview failed: {sess.name}: {error}", err=True)
    else:
        _report_preview_failures(proj, report)
    click.echo(
        "warning: the video was added, but its preview was not built (see above). "
        "Previews are only for display; build it with "
        f"`detail cache previews build --experiment {experiment_dir} "
        f"--recording {sess.name}`",
        err=True,
    )


@recording.command("backfill-kinematics")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--embedding", "embed_run_id", required=True, help="Embedding run (emb_NN)."
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Rewrite parts that already have kinematics, instead of skipping them.",
)
def recording_backfill_kinematics(
    experiment_dir: Path, recording_id: str, embed_run_id: str, overwrite: bool
) -> None:
    """Write per-frame geometry for an embedding run made before it was captured.

    Every run from now on records where the subject was on each embedded frame (§14). A run
    from before that has none, so nothing downstream can say *where* a behavior happened. This
    regenerates it from the detection zarr, which still holds every mask and box — no GPU, no
    model, no video decode. Minutes on a long recording.
    """
    proj = _open_experiment(experiment_dir)
    sess = _recording(proj, recording_id)
    bar = _StageProgress()
    try:
        written = sess.backfill_kinematics(
            embed_run_id, progress=bar, overwrite=overwrite
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    if written == 0:
        click.echo(f"{recording_id}/{embed_run_id}: already complete, nothing to do")
        return
    click.echo(f"{recording_id}/{embed_run_id}: wrote kinematics for {written} part(s)")


@recording.command("verify")
@_EXPERIMENT_OPT
@_RECORDING_OPT
def recording_verify(experiment_dir: Path, recording_id: str) -> None:
    """Check a recording's video parts are present and unchanged.

    Referenced-in-place videos are compared against the fingerprint recorded when they were
    added; a moved/renamed file reports missing (relink it), a content change reports changed
    (remove and re-add). Exits non-zero if anything is wrong.
    """
    from detail.storage import recording as recording_store

    proj = _open_experiment(experiment_dir)
    rec = _recording(proj, recording_id)
    if not recording_store.video_files(recording_store.read_recording(rec.dir)):
        click.echo(f"{recording_id}: no video yet, so there is nothing to check")
        return
    issues = rec.verify_videos()
    if not issues:
        click.echo(f"{recording_id}: all video parts present and unchanged")
        return
    for issue in issues:
        click.echo(f"  ✗ part {issue.part_id} [{issue.kind}]: {issue.detail}")
    raise SystemExit(1)


@recording.command("relink")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--part",
    "part_index",
    required=True,
    type=int,
    help="0-based part index to relink.",
)
@click.option(
    "--video",
    "video_path",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="The video file. For a copied part, defaults to the file it was copied from "
    "when that is on this machine.",
)
def recording_relink(
    experiment_dir: Path, recording_id: str, part_index: int, video_path: Path | None
) -> None:
    """Point a video part at its file: a moved referenced video, or a copied one to put back.

    A referenced part is pointed at its moved or renamed file, accepted only if its content
    matches what was recorded. For a whole moved library, re-point the root instead (`detail
    video-root set`). If the part's video root is not registered on this machine, relinking
    one file registers it, which relinks every recording under that root.

    A copied part whose video is not here (an experiment imported without its video) gets the
    file copied back in, accepted only if its frame times and resolution match the recorded
    part's exactly. Without --video it uses the file the part was copied from, when that is
    on this machine. Shows a bar while it checks and copies.
    """
    proj = _open_experiment(experiment_dir)
    rec = _recording(proj, recording_id)
    part = next((p for p in rec.parts_to_locate() if p.index == part_index), None)
    copied = part is not None and part.copied
    if video_path is None:
        if part is not None and part.copied_from is not None:
            video_path = part.copied_from
            click.echo(f"using the file it was copied from: {video_path}")
        else:
            raise click.ClickException(
                "--video is required: "
                + (
                    "the file this part was copied from is not on this machine"
                    if copied
                    else "name the video file to point this part at"
                )
            )
    bar = _StageProgress()
    try:
        registered = rec.relink_video_part(part_index, video_path, progress=bar)
    except (FileNotFoundError, IndexError, ValueError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    if copied:
        click.echo(
            f"put back {recording_id} part {part_index}: copied in from {video_path}"
        )
    else:
        click.echo(f"relinked {recording_id} part {part_index} -> {video_path}")
    if registered is not None:
        from detail.storage import video_roots

        click.echo(
            f"registered video root {registered!r} -> "
            f"{video_roots.video_root(registered)}; other recordings under it "
            "resolve too"
        )


@recording.command("add-stream")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option("--name", required=True, type=str, help="Stream name (e.g. medpc).")
@click.option(
    "--importer",
    default=None,
    type=str,
    help="Importer id (e.g. medpc). Omit when --preset names one.",
)
@click.option(
    "--data",
    "data_path",
    required=True,
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Primary stream data file to import.",
)
@click.option(
    "--program",
    "program_path",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="MED-PC .MPC program (the 'program' aux input for the medpc importer).",
)
@click.option(
    "--covered-behaviors",
    "covered_behaviors",
    type=str,
    help="Behavior stream only: comma-separated scored-behavior vocabulary "
    "(declares behaviors scored but never observed). Defaults to labels present.",
)
@click.option(
    "--origin",
    type=click.Choice(sorted(streams_store.BEHAVIOR_ORIGINS)),
    help="Behavior stream only: whether the labels are a human's scoring "
    "(ground_truth) or a method's output (predicted). Defaults to the file's own "
    "origin column, else ground_truth. A predicted stream is never read as ground "
    "truth, so a method cannot be scored against its own labels.",
)
@click.option(
    "--time-column",
    "time_column",
    type=str,
    help="CSV importers: the header name of the column carrying time. Omit to auto-detect "
    "a column named seconds, frame or t_ns.",
)
@click.option(
    "--time-is",
    "time_kind",
    type=click.Choice(["auto", "frame", "seconds", "t_ns", "utc"]),
    default="auto",
    show_default=True,
    help="CSV importers: what the time column holds. 'auto' reads it from the column's name.",
)
@click.option(
    "--channel",
    "channels",
    multiple=True,
    metavar="COLUMN[=NAME]",
    help="CSV importers: bring in this column as a channel, optionally renamed. Repeat for "
    "several. Omit for every column except the time column. One CSV makes one stream "
    "carrying the channels named here.",
)
@click.option(
    "--medpc-subject",
    "medpc_subject",
    type=str,
    help="MED-PC, a file holding several sessions: " + _MEDPC["subject"].description,
)
@click.option(
    "--medpc-box",
    "medpc_box",
    type=str,
    help="MED-PC, a file holding several sessions: " + _MEDPC["box"].description,
)
@click.option(
    "--medpc-session",
    "medpc_session",
    type=click.IntRange(min=1),
    help="MED-PC, a file holding several sessions: " + _MEDPC["session"].description,
)
@click.option(
    "--boris-observation",
    "boris_observation",
    type=str,
    help="BORIS: " + _BORIS["observation"].description,
)
@click.option(
    "--boris-subject",
    "boris_subject",
    type=str,
    help="BORIS: " + _BORIS["subject"].description,
)
@click.option(
    "--offset",
    "offset_s",
    type=FINITE_FLOAT,
    default=None,
    help="Alignment offset in seconds, set at import: positive if the stream began after "
    "the video. Only for an external-clock stream; edit later with set-stream-offset. "
    "Default 0, or with --preset the preset's own offset.",
)
@click.option(
    "--preset",
    default=None,
    type=str,
    help="Use a saved preset's importer + config instead of --importer and the mapping "
    "flags (--time-column, --time-is, --channel), which are refused beside it. See "
    "`experiment list-stream-presets`.",
)
@click.option(
    "--save-preset",
    "save_preset",
    default=None,
    type=str,
    help="Save this import's importer + mapping under this name, for reuse on the "
    "experiment's other recordings. Saved only if the import succeeds; a name already "
    "taken is refused before anything is imported.",
)
def recording_add_stream(
    experiment_dir: Path,
    recording_id: str,
    name: str,
    importer: str | None,
    data_path: Path,
    program_path: Path | None,
    covered_behaviors: str | None,
    origin: str | None,
    time_column: str | None,
    time_kind: str,
    channels: tuple[str, ...],
    medpc_subject: str | None,
    medpc_box: str | None,
    medpc_session: int | None,
    boris_observation: str | None,
    boris_subject: str | None,
    offset_s: float | None,
    preset: str | None,
    save_preset: str | None,
) -> None:
    """Import a time-series stream into a recording (Section 3).

    A MED-PC file holding several sessions, or a BORIS export holding several observations or
    subjects, is refused until the --medpc-* / --boris-* flags pick one; the refusal lists
    what the file holds. They are never merged.
    """
    # A preset carries the importer, and the branch below reads it back off the preset —
    # so `--importer` is not merely redundant there, it is ignored. It was `required=True`
    # while `--preset`'s own help said "instead of --importer", which made the documented
    # preset form unrunnable: click refused it before the body could branch.
    if (importer is None) == (preset is None):
        raise click.ClickException("pass exactly one of --importer or --preset")
    if preset is not None:
        # The preset carries the mapping, and the branch below never reads these: refused
        # rather than dropped, so a `--channel` meant to narrow the import is not ignored.
        mapping = [
            flag
            for flag, given in (
                ("--time-column", time_column is not None),
                ("--time-is", time_kind != "auto"),
                ("--channel", bool(channels)),
                ("--covered-behaviors", covered_behaviors is not None),
            )
            if given
        ]
        if mapping:
            raise UserInputError(
                f"{', '.join(mapping)} cannot be used with --preset, which carries its own "
                "mapping; drop them, or pass --importer with the flags instead"
            )
    proj = _open_experiment(experiment_dir)
    aux = {"program": program_path} if program_path is not None else None
    covered = (
        [b.strip() for b in covered_behaviors.split(",") if b.strip()]
        if covered_behaviors
        else None
    )
    config = _stream_config(time_column, time_kind, channels)
    selection = _selection_config(
        importer if preset is None else None,
        medpc_subject=medpc_subject,
        medpc_box=medpc_box,
        medpc_session=medpc_session,
        boris_observation=boris_observation,
        boris_subject=boris_subject,
    )
    # `None` with --preset means "the preset's own offset"; without one it is 0.
    offset_ns = None if offset_s is None else _option_t_ns(offset_s, "--offset")
    try:
        sess = _recording(proj, recording_id)
        if save_preset is not None:
            # The preset is saved after the import, but its name is checked before: a
            # taken name failed the command with the stream already added (#29).
            proj.check_new_stream_preset_name(save_preset)
        if preset is not None:
            sess.add_stream_from_preset(
                name,
                data_path,
                preset,
                offset_ns=offset_ns,
                aux_paths=aux,
                origin=origin,
            )
            saved = proj.get_stream_preset(preset)
            importer = saved["importer"]
            config = saved.get("config") or None
        else:
            sess.add_stream(
                name,
                raw_path=data_path,
                importer=importer,
                aux_paths=aux,
                covered_behaviors=covered,
                origin=origin,
                config=selection if selection is not None else config,
                offset_ns=0 if offset_ns is None else offset_ns,
            )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    # Saved after the import, never before: a preset is worth keeping once the mapping
    # has been shown to read the file, and a saved-but-broken one is worse than none.
    # Only the mapping: which session or subject this file held is about this file, and
    # the same choice on the next recording's file would pick another animal's data.
    if save_preset is not None:
        try:
            proj.save_stream_preset(save_preset, importer, config)
        except ValueError as e:
            raise click.ClickException(
                f"added stream {name!r} to {recording_id}, but the preset was not saved: {e}"
            )
    events = sess.stream_events(name)
    entry = next((s for s in sess.streams if s.get("name") == name), {})
    marker = (
        f" [{streams_store.stream_origin(entry)}]"
        if entry.get("role") == "behavior"
        else ""
    )
    click.echo(
        f"added stream {name!r} ({importer}) to {recording_id}: {len(events)} rows{marker}"
    )
    for note in (entry.get("metadata") or {}).get("notes") or []:
        click.echo(f"  note: {note}")


@experiment.command("list-stream-presets")
@_EXPERIMENT_OPT
def experiment_list_stream_presets(experiment_dir: Path) -> None:
    """List saved stream-import presets (Section 3)."""
    proj = _open_experiment(experiment_dir)
    presets = proj.list_stream_presets()
    if not presets:
        click.echo("no saved stream presets")
        return
    for preset in presets:
        config = preset.get("config") or {}
        channels = ", ".join(c["name"] for c in config.get("channels") or []) or "all"
        click.echo(
            f"{preset['name']}  importer={preset['importer']}  "
            f"time={config.get('time_column', 'auto')} "
            f"({config.get('time_kind', 'auto')})  channels: {channels}"
        )


@experiment.command("delete-stream-preset")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
def experiment_delete_stream_preset(experiment_dir: Path, name: str) -> None:
    """Delete a saved stream-import preset."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.delete_stream_preset(name)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted stream preset: {name}")


@experiment.command("export-stream-preset")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Write the preset to this file.",
)
def experiment_export_stream_preset(
    experiment_dir: Path, name: str, out_path: Path
) -> None:
    """Write a saved stream preset to a standalone, shareable YAML file."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.export_stream_preset(name, out_path)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"wrote {out_path}")


@experiment.command("import-stream-preset")
@_EXPERIMENT_OPT
@click.argument("path", type=UserPath(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--name", default=None, type=str, help="Save under this name instead of the file's."
)
@click.option(
    "--replace", is_flag=True, help="Overwrite an existing preset with the same name."
)
def experiment_import_stream_preset(
    experiment_dir: Path, path: Path, name: str | None, replace: bool
) -> None:
    """Import a stream preset from a standalone YAML file (Section 3)."""
    proj = _open_experiment(experiment_dir)
    try:
        saved = proj.import_stream_preset(path, name=name, overwrite=replace)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"imported stream preset: {saved}")


@experiment.command("list-headings")
@_EXPERIMENT_OPT
def experiment_list_headings(experiment_dir: Path) -> None:
    """List saved heading definitions (§14 egocentric alignment)."""
    proj = _open_experiment(experiment_dir)
    definitions = proj.heading_definitions()
    if not definitions:
        click.echo("no saved heading definitions")
        return
    for definition in definitions:
        click.echo(
            f"{definition['name']}  front: {', '.join(definition['front'])}  "
            f"back: {', '.join(definition['back'])}"
        )


@experiment.command("save-heading")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
@click.option(
    "--front",
    "front",
    required=True,
    type=str,
    help="Comma-separated skeleton point names whose mean is the front.",
)
@click.option(
    "--back",
    "back",
    required=True,
    type=str,
    help="Comma-separated skeleton point names whose mean is the back.",
)
@click.option(
    "--replace", is_flag=True, help="Overwrite an existing definition with this name."
)
def experiment_save_heading(
    experiment_dir: Path, name: str, front: str, back: str, replace: bool
) -> None:
    """Save a heading definition: the heading points from the back set's mean to the
    front set's mean. Checked against a skeleton when an embedding run uses it."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.save_heading_definition(
            name,
            [p.strip() for p in front.split(",") if p.strip()],
            [p.strip() for p in back.split(",") if p.strip()],
            overwrite=replace,
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"saved heading definition: {name}")


@experiment.command("delete-heading")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
def experiment_delete_heading(experiment_dir: Path, name: str) -> None:
    """Delete a saved heading definition."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.delete_heading_definition(name)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted heading definition: {name}")


@experiment.command("export-heading")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Write the definition to this file.",
)
def experiment_export_heading(experiment_dir: Path, name: str, out_path: Path) -> None:
    """Write a saved heading definition to a standalone, shareable YAML file."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.export_heading_definition(name, out_path)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"wrote {out_path}")


@experiment.command("import-heading")
@_EXPERIMENT_OPT
@click.argument("path", type=UserPath(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--name", default=None, type=str, help="Save under this name instead of the file's."
)
@click.option(
    "--replace",
    is_flag=True,
    help="Overwrite an existing definition with the same name.",
)
def experiment_import_heading(
    experiment_dir: Path, path: Path, name: str | None, replace: bool
) -> None:
    """Import a heading definition from a standalone YAML file (§14)."""
    proj = _open_experiment(experiment_dir)
    try:
        saved = proj.import_heading_definition(path, name=name, overwrite=replace)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"imported heading definition: {saved}")


@experiment.command("apply-stream-preset")
@_EXPERIMENT_OPT
@click.option("--preset", required=True, type=str, help="Saved preset name.")
@click.option(
    "--name", required=True, type=str, help="Stream name to give each import."
)
@click.option(
    "--sources",
    "sources_path",
    required=True,
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="CSV of `recording,file[,offset_s]` — one row per recording. Paths resolve "
    "relative to this file. The offset column is per recording, which is the point: "
    "each video and logger pair has its own skew. Head the first column `video` to key "
    "rows on each recording's video file name (any case, extension optional) instead "
    "of its id or name.",
)
def experiment_apply_stream_preset(
    experiment_dir: Path, preset: str, name: str, sources_path: Path
) -> None:
    """Import one stream per recording from a saved preset (Section 3).

    The bulk form of a mapping worked out once. A recording whose file fails is reported
    and skipped, so one unreadable file does not sink the batch.
    """
    proj = _open_experiment(experiment_dir)
    try:
        by_video, sources = _read_stream_sources(sources_path)
    except ValueError as e:
        raise click.ClickException(str(e))
    counts = {"done": 0, "skipped": 0}

    def _progress(index: int, total: int, ref: str, fraction: float, status: str):
        if fraction >= 1.0:
            counts[status] = counts.get(status, 0) + 1
            click.echo(f"  [{index + 1}/{total}] {ref}: {status}")

    try:
        skipped = proj.apply_stream_preset(
            preset, sources, name=name, by_video=by_video, progress=_progress
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"imported {counts['done']}, skipped {len(skipped)}")
    for ref, error in skipped:
        click.echo(f"  skipped {ref}: {error}", err=True)
    if skipped and not counts["done"]:
        raise click.ClickException(
            "nothing was imported: every recording was skipped (reasons above)"
        )


_ONLY_OPT = click.option(
    "--only",
    "only",
    multiple=True,
    type=str,
    help="Limit to these recordings (id or name); repeatable. Default: every other "
    "recording in the experiment.",
)


def _propagate_cli(
    proj: Experiment, category_key: str, source_ref: str, only: tuple[str, ...]
) -> None:
    """Shared body for the four ``experiment propagate-*`` commands: drive
    ``Experiment.propagate_spatial`` with a tqdm bar and report what it did.

    Counts come from the result, never from the requested target list — the API drops the
    source from its own target set, so ``--only <the source>`` asks for one recording and
    copies to none. Counting the request reported ``1/1`` for that, which is a number a
    person reads and acts on.
    """
    progress = _StageProgress()
    try:
        result = proj.propagate_spatial(
            category_key, source_ref, targets=list(only) or None, progress=progress
        )
    except (KeyError, ValueError) as e:
        raise click.ClickException(str(e))
    finally:
        progress.close()
    attempted = len(result.copied) + len(result.skipped)
    click.echo(f"copied to {len(result.copied)}/{attempted} recording(s)")
    if result.flattened:
        click.echo(
            f"  camera motion dropped (copied at frame 0): {', '.join(result.flattened)}"
        )
    names = proj.recording_names()
    for rid, name, reason in result.refused:
        click.echo(f"  not copied to {names.get(rid, rid)}: {name}: {reason}", err=True)
    for rid, error in result.skipped:
        click.echo(f"  skipped {names.get(rid, rid)}: {error}", err=True)


@experiment.command("propagate-fov")
@_EXPERIMENT_OPT
@click.option(
    "--source",
    "source_ref",
    required=True,
    type=str,
    help="Recording (id or name) to copy the field of view from.",
)
@_ONLY_OPT
def experiment_propagate_fov(
    experiment_dir: Path, source_ref: str, only: tuple[str, ...]
) -> None:
    """Copy --source's field of view onto other recordings that have none (Section 10).

    Never replaces: a recording that already has a field of view is skipped and reported,
    with the tracking runs holding it named if any do. Camera motion does not travel — a
    keyframed polygon is copied at its frame-0 position."""
    proj = _open_experiment(experiment_dir)
    _propagate_cli(proj, "fov", source_ref, only)


@experiment.command("propagate-calibration")
@_EXPERIMENT_OPT
@click.option(
    "--source",
    "source_ref",
    required=True,
    type=str,
    help="Recording (id or name) to copy the distance calibration from.",
)
@_ONLY_OPT
def experiment_propagate_calibration(
    experiment_dir: Path, source_ref: str, only: tuple[str, ...]
) -> None:
    """Copy --source's distance calibration onto other recordings that have none
    (Section 10). Never replaces; a recording that already has one is skipped."""
    proj = _open_experiment(experiment_dir)
    _propagate_cli(proj, "calibration", source_ref, only)


@experiment.command("propagate-rois")
@_EXPERIMENT_OPT
@click.option(
    "--source",
    "source_ref",
    required=True,
    type=str,
    help="Recording (id or name) to copy the regions of interest from.",
)
@_ONLY_OPT
def experiment_propagate_rois(
    experiment_dir: Path, source_ref: str, only: tuple[str, ...]
) -> None:
    """Copy --source's regions of interest onto other recordings that have none
    (Section 10). Never replaces; a recording that already has any is skipped. A region
    with an active frame window is left behind — that window belongs to one recording's
    timeline."""
    proj = _open_experiment(experiment_dir)
    _propagate_cli(proj, "roi", source_ref, only)


@experiment.command("propagate-points")
@_EXPERIMENT_OPT
@click.option(
    "--source",
    "source_ref",
    required=True,
    type=str,
    help="Recording (id or name) to copy the points of interest from.",
)
@_ONLY_OPT
def experiment_propagate_points(
    experiment_dir: Path, source_ref: str, only: tuple[str, ...]
) -> None:
    """Copy --source's points of interest onto other recordings that have none
    (Section 10). Never replaces; a recording that already has any is skipped. A point
    with an active frame window is left behind."""
    proj = _open_experiment(experiment_dir)
    _propagate_cli(proj, "point", source_ref, only)


@recording.command("set-stream-offset")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option("--name", required=True, type=str, help="Stream name.")
@click.option(
    "--offset",
    "offset_s",
    required=True,
    type=FINITE_FLOAT,
    help="Alignment offset in seconds: positive if the stream began after the video.",
)
def recording_set_stream_offset(
    experiment_dir: Path, recording_id: str, name: str, offset_s: float
) -> None:
    """Set a stream's video-alignment offset (Section 3).

    `video_t_ns = stream_t_ns + offset`. The canonical data is never rewritten, so this is a
    one-number edit that can be redone at any time. The GUI's *Sync to video* finds the same
    number by scrubbing; this is how a script sets it across many recordings at once.

    Refused for a stream already on the video's clock (behavior labels, a frame-indexed CSV):
    it has no offset, and one stored would move the preview but not the scoring.
    """
    proj = _open_experiment(experiment_dir)
    try:
        sess = _recording(proj, recording_id)
        sess.set_stream_offset(name, _option_t_ns(offset_s, "--offset"))
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"{recording_id}: stream {name!r} offset set to {offset_s:.3f}s")


@recording.command("import")
@_EXPERIMENT_OPT
@click.option(
    "--bundle",
    "bundle_dir",
    required=True,
    type=UserPath(exists=True, file_okay=False, path_type=Path),
    help="Bundle folder holding a manifest.yaml + the referenced video/stream files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the manifest and print the report; import nothing.",
)
def recording_import(experiment_dir: Path, bundle_dir: Path, dry_run: bool) -> None:
    """Bulk-import recordings (video + streams + grouping variables) from a bundle
    manifest (Section 3). Validates the whole manifest before importing anything."""
    from detail.api.imports import build_specs
    from detail.storage import manifest as manifest_store

    proj = _open_experiment(experiment_dir)
    try:
        manifest = manifest_store.load_manifest(bundle_dir)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    existing = [s.id for s in proj.recordings]
    errors, warnings = manifest_store.validate_manifest(bundle_dir, manifest, existing)
    for warning in warnings:
        click.echo(f"warning: {warning}")
    if errors:
        for error in errors:
            click.echo(f"error: {error}", err=True)
        raise click.ClickException(
            f"{len(errors)} validation error(s); imported nothing"
        )
    specs = build_specs(bundle_dir, manifest)
    if dry_run:
        click.echo(
            f"manifest valid: {len(specs)} recording(s) — nothing imported (dry run)"
        )
        return
    failed = _import_with_progress(proj, specs)
    click.echo(f"imported {len(specs) - len(failed)}/{len(specs)} recording(s)")
    for sid, error in failed:
        click.echo(f"  failed {sid}: {error}", err=True)


@recording.command("import-labels")
@_EXPERIMENT_OPT
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Labels manifest (schema_version: 1; labels: {recording_id: file}).",
)
@click.option(
    "--name",
    default="behavior",
    show_default=True,
    help="Stream name created on each recording.",
)
@click.option(
    "--importer",
    default="behavior_csv",
    show_default=True,
    type=click.Choice(["behavior_csv", "boris"]),
    help="Behavior-label importer used for every file in the batch.",
)
def recording_import_labels(
    experiment_dir: Path, manifest_path: Path, name: str, importer: str
) -> None:
    """Batch-import behavior ground-truth labels onto existing recordings (Section 3).

    Adds each manifest recording's file as a ``behavior``-role stream. Validates the
    whole manifest before importing anything; a file that fails to parse is skipped
    and reported (the batch continues)."""
    proj = _open_experiment(experiment_dir)
    counts = {"total": 0, "done": 0, "skipped": 0}

    def _progress(index: int, total: int, sid: str, fraction: float, status: str):
        counts["total"] = total
        if fraction >= 1.0:
            counts[status] = counts.get(status, 0) + 1
            click.echo(f"  [{index + 1}/{total}] {sid}: {status}")

    try:
        skipped = proj.import_behavior_labels(
            manifest_path, name=name, importer=importer, progress=_progress
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(
        f"imported {counts['done']}/{counts['total']} recording(s); "
        f"skipped {len(skipped)}"
    )
    for sid, error in skipped:
        click.echo(f"  skipped {sid}: {error}", err=True)
    if skipped and not counts["done"]:
        raise click.ClickException(
            "nothing was imported: every recording was skipped (reasons above)"
        )


@recording.command("status")
@_EXPERIMENT_OPT
@_RECORDING_OPT
def recording_status(experiment_dir: Path, recording_id: str) -> None:
    """Show recording video parts, arenas, and run history."""
    proj = _open_experiment(experiment_dir)
    try:
        sess = _recording(proj, recording_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))

    from detail.storage import recording as recording_store

    data = recording_store.read_recording(sess.dir)
    video = data.get("video", {})
    click.echo(f"recording: {sess.name}  (id {sess.id})")
    for source in (src for src in sess.video_sources if src):
        click.echo(f"  source: {source}")
    click.echo(f"  video parts: {len(video.get('files', []))}")
    if "resolution" in video:
        click.echo(
            f"  resolution: {video['resolution'][0]}x{video['resolution'][1]} "
            f"@ {video.get('fps_actual')} fps"
        )
    click.echo(
        f"  arenas: {len(sess.arenas)}  ·  rois: {len(sess.rois)}  ·  "
        f"points: {len(sess.points)}"
    )
    if sess.distance_calibration:
        refusal = sess.calibration_refusal()
        if refusal is not None:
            click.echo(f"  calibration: unusable — {refusal}")
        else:
            click.echo(
                f"  calibration: {sess.pixels_per_mm:.4g} px/mm"
                f"{sess.calibration_reference()}"
            )
    for s in sess.streams:
        offset_ns = (s.get("alignment") or {}).get("offset_ns", 0)
        click.echo(
            f"  stream  : {s['name']} "
            f"({s.get('role') or s.get('shape') or s.get('type', '')}) "
            f"offset={offset_ns / 1e9:.3f}s"
        )
    for r in sess.detection_runs:
        click.echo(f"  detection: {r['id']} status={r.get('status')}")
    for r in sess.tracking_runs:
        click.echo(f"  tracking : {r['id']} status={r.get('status')}")
    for r in sess.embedding_runs:
        click.echo(f"  embedding: {r['id']} status={r.get('status')}")


@recording.command("export-spatial")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Write the recording's whole spatial setup to this CSV (round-trippable, §10).",
)
def recording_export_spatial(
    experiment_dir: Path, recording_id: str, csv_path: Path
) -> None:
    """Export a recording's spatial setup to a CSV — the field of view, the distance
    calibration, and the regions/points of interest — to share or import onto another
    recording. Import it back with import-spatial / import-fov / import-calibration."""
    proj = _open_experiment(experiment_dir)
    try:
        rec = _recording(proj, recording_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    rec.export_spatial_csv(csv_path)
    click.echo(f"wrote {csv_path}")


def _report_arrival(
    added: list[str],
    flattened: list[str],
    refused: list[tuple[str, str]],
    noun: str,
) -> None:
    """Report one *pulled* spatial import the way ``_propagate_cli`` reports a pushed one.

    Same operation, opposite direction, so the same three lines: how much arrived, whose
    camera motion was dropped on the way, and one line per definition left behind with
    its reason. A bare count is the shape this replaces, and it hid both of the last two
    — a flattened polygon and a refused name are changes to what the user asked for, and
    the run still exits 0.
    """
    attempted = len(added) + len(refused)
    click.echo(f"imported {len(added)}/{attempted} {noun}")
    if flattened:
        click.echo(
            f"  camera motion dropped (copied at frame 0): {', '.join(flattened)}"
        )
    for name, reason in refused:
        click.echo(f"  not imported: {name} {reason}", err=True)


@recording.command("import-spatial")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--from-recording",
    "from_recording_id",
    type=str,
    help="Copy ROIs + POIs from this recording id in the same experiment.",
)
@click.option(
    "--csv",
    "csv_path",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Import ROIs + POIs from an exported CSV.",
)
def recording_import_spatial(
    experiment_dir: Path,
    recording_id: str,
    from_recording_id: str | None,
    csv_path: Path | None,
) -> None:
    """Take regions + points of interest onto a recording from another recording
    (--from-recording) or a CSV (--csv) — the pull twin of experiment propagate-rois /
    propagate-points, and the same rules.

    Never replaces: a name this recording already has is refused and reported by name,
    and the ones that do not collide still arrive (regions and points are sets, so one
    collision does not refuse the rest). Camera motion does not travel — a keyframed
    polygon arrives at its frame-0 position, and the command says which ones. A region
    with an active frame window is left behind: that window belongs to one recording's
    timeline."""
    if bool(from_recording_id) == bool(csv_path):
        raise click.ClickException("give exactly one of --from-recording or --csv")
    proj = _open_experiment(experiment_dir)
    try:
        rec = _recording(proj, recording_id)
        if from_recording_id:
            added, flattened, refused = rec.import_spatial_from(
                _recording(proj, from_recording_id)
            )
        else:
            added, flattened, refused = rec.import_spatial_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    _report_arrival(added, flattened, refused, "region(s)/point(s)")


@recording.command("import-fov")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--from-recording",
    "from_recording_id",
    type=str,
    help="Copy the field of view from this recording id in the same experiment.",
)
@click.option(
    "--csv",
    "csv_path",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Import the field of view from an exported spatial CSV.",
)
def recording_import_fov(
    experiment_dir: Path,
    recording_id: str,
    from_recording_id: str | None,
    csv_path: Path | None,
) -> None:
    """Carry a field of view onto a recording from another recording (--from-recording) or
    a CSV (--csv) — recordings off one rig share it, so it need only be drawn once.

    Never replaces, and unlike the regions it is refused whole: a recording that already
    has a field of view keeps it, and the command fails saying so. Delete it on this
    recording first — along with any tracking runs holding it, which its region describes.
    Camera motion does not travel; a keyframed polygon arrives at its frame-0 position.
    """
    if bool(from_recording_id) == bool(csv_path):
        raise click.ClickException("give exactly one of --from-recording or --csv")
    proj = _open_experiment(experiment_dir)
    try:
        rec = _recording(proj, recording_id)
        if from_recording_id:
            added, flattened, refused = rec.import_field_of_view_from(
                _recording(proj, from_recording_id)
            )
        else:
            added, flattened, refused = rec.import_field_of_view_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    _report_arrival(added, flattened, refused, "field(s) of view")


@recording.command("import-calibration")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--from-recording",
    "from_recording_id",
    type=str,
    help="Copy the distance calibration from this recording id in the same experiment.",
)
@click.option(
    "--csv",
    "csv_path",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Import the distance calibration from an exported spatial CSV.",
)
def recording_import_calibration(
    experiment_dir: Path,
    recording_id: str,
    from_recording_id: str | None,
    csv_path: Path | None,
) -> None:
    """Carry a distance calibration onto a recording from another recording
    (--from-recording) or a CSV (--csv).

    Never replaces, and refused whole like the field of view: there is one pixel scale, so
    a recording that already has one keeps it and the command fails saying so. Delete it
    on this recording first (clear-calibration).

    The calibration keeps the resolution it was measured at, so it stays correct on a
    recording whose video is a different size (§10) — but it is refused onto a video of a
    different aspect ratio, where one pixels-per-mm cannot be right. Nothing to flatten or
    leave behind — it carries no frame index."""
    if bool(from_recording_id) == bool(csv_path):
        raise click.ClickException("give exactly one of --from-recording or --csv")
    proj = _open_experiment(experiment_dir)
    try:
        rec = _recording(proj, recording_id)
        if from_recording_id:
            block = rec.import_calibration_from(_recording(proj, from_recording_id))
        else:
            block = rec.import_calibration_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    w, h = block["ref_resolution"]
    click.echo(f"calibration imported: {block['pixels_per_mm']} px/mm at {w}x{h}")


@recording.command("set-calibration")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--pixels-per-mm",
    "pixels_per_mm",
    required=True,
    type=FINITE_FLOAT,
    help="Distance calibration as a pixels-per-mm scalar (the two-point pick is GUI-only).",
)
def recording_set_calibration(
    experiment_dir: Path, recording_id: str, pixels_per_mm: float
) -> None:
    """Set a recording's distance calibration from a pixels-per-mm scalar, measured at its
    current video resolution (§10). Use the GUI for the two-point pick."""
    proj = _open_experiment(experiment_dir)
    try:
        block = _recording(proj, recording_id).set_pixels_per_mm(pixels_per_mm)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    w, h = block["ref_resolution"]
    click.echo(f"calibration set: {block['pixels_per_mm']} px/mm at {w}x{h}")


@recording.command("clear-calibration")
@_EXPERIMENT_OPT
@_RECORDING_OPT
def recording_clear_calibration(experiment_dir: Path, recording_id: str) -> None:
    """Remove a recording's distance calibration (§10)."""
    proj = _open_experiment(experiment_dir)
    try:
        _recording(proj, recording_id).clear_distance_calibration()
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo("calibration cleared")


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _detection_opts(f):
    """Detection config options shared by the recording ``detection run`` and the
    experiment-level ``process detection`` commands (applied bottom-up to preserve
    the original ``--help`` order). ``--model`` takes any model id — a built-in,
    an experiment-trained, or an imported one; the source is resolved automatically.
    A pretrained (open-set) model needs ``--class`` + ``--prompt``; a local model
    is closed-set and takes neither."""
    for opt in (
        click.option(
            "--every-n-frames",
            "every_n_frames",
            default=1,
            show_default=True,
            type=_int_range(_DET_DOC["every_n_frames"]),
            help=_DET["every_n_frames"].description,
        ),
        click.option(
            "--conf",
            "conf_threshold",
            default=0.5,
            show_default=True,
            type=FINITE_FLOAT,
            help=_DET["conf_threshold"].description,
        ),
        click.option(
            "--batch-size",
            "batch_size",
            default=None,
            type=_int_range(_DET_DOC["batch_size"]),
            help=_DET["batch_size"].description,
        ),
        click.option(
            "--prompt",
            type=str,
            default=None,
            help="Text prompt for a pretrained (open-set) model; omit for a local model.",
        ),
        click.option(
            "--class",
            "class_name",
            type=str,
            default=None,
            help="Class label for a pretrained model's detections; omit for a local model.",
        ),
        click.option(
            "--variant",
            "model_variant",
            type=str,
            default=None,
            help=_DET["model_variant"].description,
        ),
        click.option(
            "--model",
            "model_id",
            required=True,
            type=str,
            help="Model id: built-in (sam3), experiment-trained (model_NNN), "
            "or imported (see `model list`). Source is resolved automatically.",
        ),
    ):
        f = opt(f)
    return f


def _detection_prompt(class_name: str | None, prompt: str | None) -> Prompt:
    """Build the detection prompt from the CLI flags. A pretrained model needs both
    ``--class`` and ``--prompt``; a local (trained/imported) model is closed-set and
    takes neither (an empty prompt). The detection validator enforces the right one
    for the resolved model."""
    if bool(class_name) != bool(prompt):
        raise click.ClickException(
            "give both --class and --prompt (a pretrained model needs them) or "
            "neither (a local model is closed-set and takes no prompt)"
        )
    if class_name and prompt:
        return Prompt(
            targets=[PromptTarget(class_name=class_name, cue=TextCue(prompt))]
        )
    return Prompt(targets=[])


def _detection_config(
    model_id: str,
    model_variant: str | None,
    class_name: str | None,
    prompt: str | None,
    conf_threshold: float,
    batch_size: int | None,
    every_n_frames: int,
    backend: str,
    devices: tuple[int, ...],
) -> dict:
    """Build the detection-module config dict (shared by recording + process runs).
    ``weights_source`` is left to ``validate`` to resolve from the model id."""
    return {
        "model_id": model_id,
        "model_variant": model_variant,
        "prompt": _detection_prompt(class_name, prompt),
        "compute": _compute_target(backend, devices),
        "conf_threshold": conf_threshold,
        "batch_size": batch_size,
        "every_n_frames": every_n_frames,
    }


@main.group()
def detection() -> None:
    """Detection (§7) — find instances of a text-prompted target across frames."""


@detection.command("run")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@_detection_opts
@_BACKEND_OPT
@_DEVICE_OPT
def detection_run(
    experiment_dir: Path,
    recording_id: str,
    model_id: str,
    model_variant: str | None,
    class_name: str | None,
    prompt: str | None,
    conf_threshold: float,
    batch_size: int | None,
    every_n_frames: int,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Submit a detection run; blocks until complete."""
    proj = _open_experiment(experiment_dir)
    sess = _recording(proj, recording_id)
    p = _detection_prompt(class_name, prompt)
    try:
        run = sess.run_detection(
            model_id=model_id,
            model_variant=model_variant,
            prompt=p,
            compute=_compute_target(backend, devices),
            conf_threshold=conf_threshold,
            batch_size=batch_size,
            every_n_frames=every_n_frames,
        )
    except (FileNotFoundError, ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted detection: {run.id}")
    _stream_progress(run, "detection")
    _check_status(run, "detection")


def _echo_quick_segment_kept(run_dir: Path, record: dict, min_conf: float) -> None:
    """What the display threshold keeps, beside the run's own all-proposals total.

    The stored summary counts every proposal, because that is what the run kept. Printing
    only that would read as "1,842 detections" for a clip holding one animal.
    """
    from detail.tools.quick_segment import (
        frame_detection_count,
        load_detections,
        sampled_confidences,
    )

    outputs = (record.get("result") or {}).get("outputs") or {}
    if "detections" in outputs:
        columns = load_detections(run_dir / outputs["detections"])
        kept = frame_detection_count(columns, min_conf)
        total = len(columns["frame_idx"])
    else:
        per_prompt = sampled_confidences(run_dir)
        kept = sum(int((c >= min_conf).sum()) for c in per_prompt.values())
        total = sum(int(c.size) for c in per_prompt.values())
    click.echo(f"  {kept} of {total} at --min-conf {min_conf}")


def _echo_prompt_test_summary(
    results: list, prompt_texts: tuple[str, ...], min_conf: float = 0.0
) -> None:
    """Per-prompt detection-rate summary across the sampled frames, at ``min_conf``.

    The run keeps every proposal, so an unfiltered count here would report the model's
    whole proposal set and a mean confidence dragged toward zero by rows nobody would
    ever look at.
    """
    if not results:
        click.echo("no frames sampled")
        return
    labels = sorted(results[0].results.keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    n_frames = len(results)
    click.echo(
        f"{n_frames} frames sampled (shown at --min-conf {min_conf}); per prompt:"
    )
    for i, label in enumerate(labels):
        kept = [
            [d for d in r.results[label].detections() if d.confidence >= min_conf]
            for r in results
        ]
        confs = [d.confidence for frame in kept for d in frame]
        frames_hit = sum(1 for frame in kept if frame)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        text = prompt_texts[i] if i < len(prompt_texts) else "?"
        click.echo(
            f'  {label} "{text}": {frames_hit}/{n_frames} frames hit, '
            f"{len(confs)} detections, mean conf {mean_conf:.3f}"
        )


@detection.command("prompt-test")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--model",
    "model_id",
    required=True,
    type=str,
    help="Detection model id (e.g. sam3).",
)
@click.option(
    "--variant",
    "model_variant",
    type=str,
    default=None,
    help="Model size/variant; omit for the default.",
)
@click.option(
    "--class",
    "class_name",
    type=str,
    default=None,
    help="Class label applied to every candidate prompt (open-vocab models).",
)
@click.option(
    "--prompt",
    "prompts",
    multiple=True,
    help="Candidate prompt text (repeatable — one per phrasing to compare). "
    "Open-vocab models only.",
)
@click.option(
    "--show-class",
    "show_classes",
    multiple=True,
    help="Trained/local model: preview only these classes (repeatable; default all).",
)
@click.option(
    "--n",
    "n_sample_frames",
    default=8,
    show_default=True,
    type=int,
    help="Random frames to sample.",
)
@_MIN_CONF_OPT
@click.option(
    "--save-as",
    "save_as",
    type=str,
    default=None,
    help="Write overlay JPGs + test.json to detections/prompt_tests/<save-as>/.",
)
@click.option(
    "--seed", type=int, default=None, help="RNG seed for reproducible sampling."
)
@_BACKEND_OPT
@_DEVICE_OPT
def detection_prompt_test(
    experiment_dir: Path,
    recording_id: str,
    model_id: str,
    model_variant: str | None,
    class_name: str | None,
    prompts: tuple[str, ...],
    show_classes: tuple[str, ...],
    n_sample_frames: int,
    min_conf: float,
    save_as: str | None,
    seed: int | None,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Preview a detection model on a few sampled frames (Section 8) — a fast QC loop
    before a full run. An open-vocab model compares each --prompt candidate for
    --class; a trained/local model previews its own classes (filter with --show-class,
    one result column per class).

    The model runs wide open and keeps every proposal; --min-conf chooses what the
    summary counts and what the --save-as overlays draw."""
    if (class_name or prompts) and show_classes:
        raise click.ClickException(
            "give text prompts (--class/--prompt, open-vocab) or --show-class "
            "(trained model), not both"
        )
    if bool(class_name) != bool(prompts):
        raise click.ClickException(
            "give both --class and --prompt (open-vocab), or neither (a trained "
            "model is closed-set — optionally narrow it with --show-class)"
        )
    proj = _open_experiment(experiment_dir)
    sess = _recording(proj, recording_id)
    prompt_objs = [
        Prompt(targets=[PromptTarget(class_name=class_name, cue=TextCue(text))])
        for text in prompts
    ]
    try:
        results = sess.run_prompt_test(
            model_id=model_id,
            prompts=prompt_objs,
            display_classes=list(show_classes) or None,
            n_sample_frames=n_sample_frames,
            compute=_compute_target(backend, devices),
            save_as=save_as,
            seed=seed,
            model_variant=model_variant,
            min_conf=min_conf,
        )
    except (ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    _echo_prompt_test_summary(results, prompts or show_classes, min_conf)
    if save_as is not None:
        out = sess.dir / "detections" / "prompt_tests" / save_as
        click.echo(f"saved overlays + test.json to: {out}")


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
def _coerce_hyperparam(schema: dict, key: str, raw: str):
    """Coerce a CLI ``--hyperparam KEY=VALUE`` string to the type the selected
    tracker declares in ``HYPERPARAMS_SCHEMA`` — so the generic flag works for any
    tracker, present or future, the same way the GUI's schema-driven form does."""
    if key not in schema:
        raise click.ClickException(
            f"unknown hyperparam {key!r}; allowed: {sorted(schema)}"
        )
    typ = schema[key][0]
    if typ is bool:
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise click.ClickException(
            f"--hyperparam {key}: cannot read {raw!r} as bool (use true/false)"
        )
    try:
        return typ(raw)
    except (ValueError, TypeError):
        raise click.ClickException(
            f"--hyperparam {key}: cannot read {raw!r} as {typ.__name__}"
        )


def _tracking_opts(f):
    """Tracking config options shared by the recording ``tracking run`` and the
    experiment-level ``process tracking`` commands (applied bottom-up to preserve the
    original ``--help`` order)."""
    for opt in (
        click.option(
            "--arena",
            "arena_ref",
            type=str,
            help=(
                "Field of view to track inside, by **name** (recommended) or by this "
                "recording's own arena id. A name is the same field of view on every "
                "recording; an id is minted per recording, so one id names a different "
                "polygon on the recording next to it."
            ),
        ),
        click.option(
            "--every-n-frames",
            "every_n_frames",
            default=None,
            type=int,
            help=_TRK["every_n_frames"].description,
        ),
        click.option(
            "--hyperparam",
            "raw_hyperparams",
            multiple=True,
            metavar="KEY=VALUE",
            help="Set any hyperparam of the selected tracker (repeatable), e.g. "
            "--hyperparam max_secondary=3. Coerced to the type the tracker declares; "
            "see the tracker's schema. For knobs without a dedicated flag above.",
        ),
        click.option(
            "--require-contiguous/--no-require-contiguous",
            default=None,
            help="anchor_tracker: reject masks with more than one connected component.",
        ),
        click.option(
            "--max-secondary",
            type=int,
            help="anchor_tracker: max secondary detections kept per anchor.",
        ),
        click.option(
            "--secondary-conf",
            type=FINITE_FLOAT,
            help="anchor_tracker: min confidence for overlapping secondary detections.",
        ),
        click.option(
            "--num-subjects",
            type=int,
            help="anchor_tracker: subjects to keep per frame.",
        ),
        click.option(
            "--conf-threshold",
            type=FINITE_FLOAT,
            help="Confidence filter applied by the tracker.",
        ),
        click.option(
            "--max-missed", type=int, help="Max missed frames before a track dies."
        ),
        click.option(
            "--iou-threshold",
            type=FINITE_FLOAT,
            help="Box-overlap threshold. What it gates depends on the tracker: how much "
            "a fragment must overlap the subject to be merged into it.",
        ),
        click.option(
            "--tracker",
            "tracker_id",
            default=_TRK["tracker_id"].default,
            show_default=True,
            type=str,
            help=_TRK["tracker_id"].description,
        ),
        click.option(
            "--detection",
            "detection_run_id",
            default=None,
            type=str,
            help=_TRK["detection_run_id"].description,
        ),
    ):
        f = opt(f)
    return f


def _parse_run_ref(ref: str) -> tuple[str, str]:
    """Split a ``RECORDING:run_id`` upstream reference for the ``--*-ref`` selectors."""
    sid, _, run_id = ref.partition(":")
    if not sid or not run_id:
        raise click.ClickException(
            f"expected RECORDING:run_id (e.g. s01:det_02), got {ref!r}"
        )
    return sid, run_id


def _spec_from_ref(proj, module_id: str, ref: str) -> dict:
    """Derive an upstream spec from one named run (``RECORDING:run_id``) — the CLI
    counterpart to the GUI spec pickers (§6.5); each recording resolves its own match.
    """
    from detail.api import run_spec

    sid, run_id = _parse_run_ref(ref)
    try:
        return run_spec.spec_from_run(proj.get_recording(sid), module_id, run_id)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


#: ``--embed-ref`` — the embedding counterpart of ``--tracking-ref``, for an analysis that reads
#: one embedding run per recording. Hand-written like ``--tracking-ref``: the spec it names is a
#: selection, not a field of the config model.
_EMBED_REF_OPT = click.option(
    "--embed-ref", "embed_refs", multiple=True, metavar="RECORDING:emb_NN",
    help="Use the embedding settings of this run, resolved to each recording's own run with "
    "those settings, whatever its id (alternative to --embed-run). Repeat it to pool "
    "recordings whose embeddings were made through different chains (e.g. two detection "
    "models): each recording reads its run of the one named setting it has, and one with "
    "none or several is refused. Left out along with --embed-run, the one set of settings "
    "every recording has a run of is used.",
)  # fmt: skip


def _embedding_selection(
    proj, embed_run_id: str | None, embed_refs: tuple[str, ...]
) -> dict:
    """The config fields naming an analysis's embedding runs (§6.5) — ``--embed-run`` is one
    id in every recording, ``--embed-ref`` is settings resolved per recording (repeated, several
    settings each recording picks one of), and neither leaves the module to find the one set of
    settings every recording shares."""
    if embed_run_id and embed_refs:
        raise click.ClickException("pass at most one of --embed-run or --embed-ref")
    return _spec_selection(proj, "embedding", embed_refs) or (
        {"embed_run_id": embed_run_id} if embed_run_id else {}
    )


def _spec_selection(proj, module_id: str, refs: tuple[str, ...]) -> dict:
    """``<module>_spec`` for one ``RECORDING:run`` ref, ``<module>_specs`` for several (§6.5),
    nothing for none — the one place a repeated ``--*-ref`` becomes a selection."""
    specs = [_spec_from_ref(proj, module_id, ref) for ref in refs]
    if len(specs) == 1:
        return {f"{module_id}_spec": specs[0]}
    return {f"{module_id}_specs": specs} if specs else {}


#: The two spellings trackers use for "frames the subject may be absent before the track
#: ends". One flag, because it is one idea: a tracker may call it `max_missed_frames` or
#: `max_missed` (the single-subject trackers), and a flag hardcoded to either spelling is a
#: flag that fails on the other — offered on a command whose `--tracker` reaches both.
_MISSED_KEYS = ("max_missed_frames", "max_missed")


def _surfaced_trackers() -> list[str]:
    """The §13 tracker ids this build offers, for the "available" list a user is shown.

    Filtered through §25 **here** — where the choices are enumerated — and never at the
    `TRACKERS` lookup that resolves one, so a hidden tracker named explicitly still runs and
    a saved run that used it still opens (hiding governs launching, not reading).
    """
    import detail.pipeline.tracking  # noqa: F401 — registers TRACKERS

    from detail.core.registries import TRACKERS

    return surfaces.surfaced(sorted(TRACKERS))


def _missed_key(tracker_id: str) -> str:
    """Whichever of ``_MISSED_KEYS`` the chosen tracker declares.

    Falls back to the first so an unknown tracker still reaches the schema check below and
    fails there, naming what it does accept, rather than here with a key error.
    """
    import detail.pipeline.tracking  # noqa: F401 — registers TRACKERS

    from detail.core.registries import TRACKERS

    schema = getattr(TRACKERS.get(tracker_id), "HYPERPARAMS_SCHEMA", {})
    return next((key for key in _MISSED_KEYS if key in schema), _MISSED_KEYS[0])


def _build_tracking_hyperparams(
    tracker_id: str,
    iou_threshold: float | None,
    max_missed: int | None,
    conf_threshold: float | None,
    num_subjects: int | None,
    secondary_conf: float | None,
    max_secondary: int | None,
    require_contiguous: bool | None,
    raw_hyperparams: tuple[str, ...],
) -> dict:
    """Collect dedicated flags + ``--hyperparam KEY=VALUE`` into the tracker's
    hyperparam dict, coerced against its schema (shared by recording + process runs)."""
    hyperparams: dict = {}
    if iou_threshold is not None:
        hyperparams["iou_threshold"] = iou_threshold
    if max_missed is not None:
        hyperparams[_missed_key(tracker_id)] = max_missed
    if conf_threshold is not None:
        hyperparams["conf_threshold"] = conf_threshold
    if num_subjects is not None:
        hyperparams["num_subjects"] = num_subjects
    if secondary_conf is not None:
        hyperparams["secondary_conf"] = secondary_conf
    if max_secondary is not None:
        hyperparams["max_secondary"] = max_secondary
    if require_contiguous is not None:
        hyperparams["require_contiguous"] = require_contiguous
    if raw_hyperparams:
        import detail.pipeline.tracking  # noqa: F401 — registers TRACKERS

        from detail.core.registries import TRACKERS

        tracker_cls = TRACKERS.get(tracker_id)
        if tracker_cls is None:
            raise click.ClickException(
                f"unknown tracker {tracker_id!r}; available: {_surfaced_trackers()}"
            )
        schema = dict(getattr(tracker_cls, "HYPERPARAMS_SCHEMA", {}))
        for item in raw_hyperparams:
            if "=" not in item:
                raise click.ClickException(
                    f"--hyperparam must be KEY=VALUE, got {item!r}"
                )
            key, raw = item.split("=", 1)
            key = key.strip()
            if key in hyperparams:
                raise click.ClickException(
                    f"hyperparam {key!r} set twice (a dedicated flag and --hyperparam)"
                )
            hyperparams[key] = _coerce_hyperparam(schema, key, raw)
    return hyperparams


@main.group()
def tracking() -> None:
    """Tracking (§13) — link detections frame-to-frame into tracks."""


@tracking.command("run")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@_tracking_opts
def tracking_run(
    experiment_dir: Path,
    recording_id: str,
    detection_run_id: str,
    tracker_id: str,
    iou_threshold: float | None,
    max_missed: int | None,
    conf_threshold: float | None,
    num_subjects: int | None,
    secondary_conf: float | None,
    max_secondary: int | None,
    require_contiguous: bool | None,
    raw_hyperparams: tuple[str, ...],
    arena_ref: str | None,
    every_n_frames: int | None,
) -> None:
    """Submit a tracking run; blocks until complete."""
    if (
        not detection_run_id
    ):  # optional in the shared opts (for process --detection-ref)
        raise click.ClickException("Missing option '--detection'.")
    proj = _open_experiment(experiment_dir)
    sess = _recording(proj, recording_id)
    hyperparams = _build_tracking_hyperparams(
        tracker_id,
        iou_threshold,
        max_missed,
        conf_threshold,
        num_subjects,
        secondary_conf,
        max_secondary,
        require_contiguous,
        raw_hyperparams,
    )
    try:
        run = sess.run_tracking(
            detection_run_id=detection_run_id,
            tracker_id=tracker_id,
            hyperparams=hyperparams,
            arena_name=arena_ref,
            every_n_frames=every_n_frames,
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted tracking: {run.id}")
    _stream_progress(run, "tracking")
    _check_status(run, "tracking")


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
@main.group()
def embedding() -> None:
    """Embedding (§14) — embed each tracked detection into a feature vector."""


@embedding.command("run")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--tracking",
    "tracking_run_id",
    required=True,
    type=str,
    help=_EMB["tracking_run_id"].description,
)
@click.option(
    "--tracks",
    "track_ids",
    default=None,
    type=str,
    help="Comma-separated track ids to embed (e.g. 0 or 0,1). "
    "Mutually exclusive with --all-tracks.",
)
@click.option(
    "--all-tracks",
    "all_tracks",
    is_flag=True,
    default=False,
    help="Embed every track in the run (mutually exclusive with --tracks).",
)
@click.option(
    "--model",
    "model_id",
    default=_EMB["model_id"].default,
    show_default=True,
    type=str,
    help=_EMB["model_id"].description,
)
@click.option(
    "--variant",
    "model_variant",
    type=str,
    default=None,
    help=_EMB["model_variant"].description,
)
@click.option(
    "--input-size",
    "input_size",
    default=None,
    type=int,
    help=_EMB["input_size"].description,
)
@click.option(
    "--readout",
    "readout",
    default=None,
    type=str,
    help=_EMB["readout"].description,
)
@click.option(
    "--resize",
    "renderer_size",
    default=None,
    type=int,
    help=_EMB["resize"].description,
)
@click.option(
    "--constructor",
    "constructor",
    type=click.Choice(surfaced_constructors()),
    default=None,
    help=_EMB["constructor"].description,
)
@click.option(
    "--batch-size",
    "batch_size",
    type=_int_range(_EMB_DOC["batch_size"]),
    default=32,
    show_default=True,
    help=_EMB["batch_size"].description,
)
@click.option(
    "--every-n-frames",
    "every_n_frames",
    default=None,
    type=int,
    help=_EMB["every_n_frames"].description,
)
@_ALIGN_OPTS
@_BACKEND_OPT
@_DEVICE_OPT
def embedding_run(
    experiment_dir: Path,
    recording_id: str,
    tracking_run_id: str,
    track_ids: str | None,
    all_tracks: bool,
    model_id: str,
    model_variant: str | None,
    input_size: int | None,
    readout: str | None,
    renderer_size: int | None,
    constructor: str | None,
    batch_size: int,
    every_n_frames: int | None,
    align: str,
    keypoint_run_id: str | None,
    heading: str | None,
    max_gap_s: float | None,
    mirror_hold_s: float | None,
    facing_camera_ratio: float | None,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Submit an embedding run; blocks until complete. ``--align rotate|mirror`` with
    ``--keypoint-run kt_NN --heading NAME`` turns each animal to face right (§14).
    """
    proj = _open_experiment(experiment_dir)
    sess = _recording(proj, recording_id)
    if all_tracks and track_ids:
        raise click.ClickException("use either --tracks or --all-tracks, not both")
    if not all_tracks and not track_ids:
        raise click.ClickException("provide --tracks or --all-tracks")
    if not all_tracks:
        try:
            ids = [int(t.strip()) for t in track_ids.split(",") if t.strip()]
        except ValueError:
            raise click.ClickException(
                f"--tracks must be comma-separated ints, got {track_ids!r}"
            )
    try:
        if all_tracks:
            ids = sess.track_ids(tracking_run_id)
        # One resolution path for every case: a named constructor (DEFAULT_CONSTRUCTOR
        # when omitted) resolved per recording, with `--resize` omitted (None) deferring
        # to the model's own input size — the same target the GUI and API resolve. An
        # aligned run leaves the spec to `validate`, which sizes the canvas from the
        # turned masks and inserts the alignment step (§14).
        renderer_spec = (
            None
            if align != "none"
            else sess.resolve_constructor(
                constructor or DEFAULT_CONSTRUCTOR,
                tracking_run_id,
                ids,
                model_id=model_id,
                resize=renderer_size,
                input_size=input_size,
            )
        )
        alignment = _alignment_config(
            align, heading, max_gap_s, mirror_hold_s, facing_camera_ratio
        )
        run = sess.run_embedding(
            tracking_run_id=tracking_run_id,
            track_ids=ids,
            model_id=model_id,
            model_variant=model_variant,
            renderer_spec=renderer_spec,
            constructor=constructor or DEFAULT_CONSTRUCTOR,
            readout=readout,
            input_size=input_size,
            batch_size=batch_size,
            every_n_frames=every_n_frames,
            compute=_compute_target(backend, devices),
            align=alignment["align"],
            keypoint_run_id=keypoint_run_id,
            heading=alignment["heading"],
            max_gap_s=alignment["max_gap_s"],
            mirror_hold_s=alignment["mirror_hold_s"],
            facing_camera_ratio=alignment["facing_camera_ratio"],
        )
    except (ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted embedding: {run.id}")
    _stream_progress(run, "embedding")
    _check_status(run, "embedding")


# --------------------------------------------------------------------------- #
# process (experiment-level processing runs, §15)
# --------------------------------------------------------------------------- #
@main.group()
def process() -> None:
    """Experiment-level processing runs (§15) — run a stage across one or more
    recordings, recorded as ``run_NNN`` (the source a dataset is built from)."""


@process.command("detection")
@_EXPERIMENT_OPT
@_RECORDINGS_OPT
@_detection_opts
@_BACKEND_OPT
@_DEVICE_OPT
def process_detection(
    experiment_dir: Path,
    recording_ids: tuple[str, ...],
    model_id: str,
    model_variant: str | None,
    class_name: str | None,
    prompt: str | None,
    conf_threshold: float,
    batch_size: int | None,
    every_n_frames: int,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Run detection across the selected recordings; writes an experiment run (run_NNN)."""
    proj = _open_experiment(experiment_dir)
    config = _detection_config(
        model_id,
        model_variant,
        class_name,
        prompt,
        conf_threshold,
        batch_size,
        every_n_frames,
        backend,
        devices,
    )
    try:
        refs = _resolve_recording_refs(proj, recording_ids)
        run = _with_stage_progress(
            lambda bar: proj.run_processing("detection", refs, config, progress=bar)
        )
    except (FileNotFoundError, ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted processing run: {run.id}")
    _stream_processing_progress(run, "detection")
    _check_status(run, "processing")


@process.command("tracking")
@_EXPERIMENT_OPT
@_RECORDINGS_OPT
@_tracking_opts
@click.option(
    "--detection-ref", "detection_ref", default=None,
    help="RECORDING:det_NN — derive the detection kind from this run and resolve each "
    "recording's own match (alternative to --detection, for recordings with divergent ids).",
)  # fmt: skip
def process_tracking(
    experiment_dir: Path,
    recording_ids: tuple[str, ...],
    detection_run_id: str | None,
    tracker_id: str,
    iou_threshold: float | None,
    max_missed: int | None,
    conf_threshold: float | None,
    num_subjects: int | None,
    secondary_conf: float | None,
    max_secondary: int | None,
    require_contiguous: bool | None,
    raw_hyperparams: tuple[str, ...],
    arena_ref: str | None,
    every_n_frames: int | None,
    detection_ref: str | None,
) -> None:
    """Run tracking across the selected recordings; writes an experiment run (run_NNN).

    Point at the detections with either ``--detection det_NN`` (that id must be
    complete in every selected recording) or ``--detection-ref RECORDING:det_NN`` (derive
    the detection kind from one run and resolve each recording's own match)."""
    if bool(detection_run_id) == bool(detection_ref):
        raise click.ClickException("pass exactly one of --detection or --detection-ref")
    proj = _open_experiment(experiment_dir)
    hyperparams = _build_tracking_hyperparams(
        tracker_id,
        iou_threshold,
        max_missed,
        conf_threshold,
        num_subjects,
        secondary_conf,
        max_secondary,
        require_contiguous,
        raw_hyperparams,
    )
    config = {
        "tracker_id": tracker_id,
        "hyperparams": hyperparams,
        # A **name**, resolved to each recording's own arena id in `validate()`. Passing an
        # id here had the GUI's bug exactly: one string applied across every selected
        # recording, where the same id is a different polygon on each of them.
        "arena_name": arena_ref,
        "every_n_frames": every_n_frames,
    }
    if detection_ref:
        config["detection_spec"] = _spec_from_ref(proj, "detection", detection_ref)
    else:
        config["detection_run_id"] = detection_run_id
    try:
        refs = _resolve_recording_refs(proj, recording_ids)
        run = _with_stage_progress(
            lambda bar: proj.run_processing("tracking", refs, config, progress=bar)
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted processing run: {run.id}")
    _stream_processing_progress(run, "tracking")
    _check_status(run, "processing")


@process.command("embedding")
@_EXPERIMENT_OPT
@_RECORDINGS_OPT
@click.option(
    "--tracking",
    "tracking_run_id",
    default=None,
    type=str,
    help="Tracking run id, complete in every selected recording.",
)
@click.option(
    "--tracking-ref",
    "tracking_ref",
    default=None,
    help="RECORDING:track_NN — derive the tracking kind from this run and resolve each "
    "recording's own match (alternative to --tracking, for recordings with divergent ids).",
)
@click.option(
    "--tracks",
    "track_ids",
    default=None,
    type=str,
    help="Comma-separated track ids to embed. Mutually exclusive with --all-tracks.",
)
@click.option(
    "--all-tracks",
    "all_tracks",
    is_flag=True,
    default=False,
    help="Embed every track in each recording (mutually exclusive with --tracks).",
)
@click.option(
    "--model",
    "model_id",
    default=_EMB["model_id"].default,
    show_default=True,
    type=str,
    help=_EMB["model_id"].description,
)
@click.option(
    "--variant",
    "model_variant",
    type=str,
    default=None,
    help=_EMB["model_variant"].description,
)
@click.option(
    "--input-size",
    "input_size",
    default=None,
    type=int,
    help=_EMB["input_size"].description,
)
@click.option(
    "--readout",
    "readout",
    default=None,
    type=str,
    help=_EMB["readout"].description,
)
@click.option(
    "--resize",
    "renderer_size",
    default=None,
    type=int,
    help=_EMB["resize"].description,
)
@click.option(
    "--constructor",
    "constructor",
    type=click.Choice(surfaced_constructors()),
    default=None,
    help=_EMB["constructor"].description,
)
@click.option(
    "--canvas-scope",
    "canvas_scope",
    type=click.Choice(["per_recording", "shared"]),
    default="per_recording",
    show_default=True,
    help=_EMB["canvas_scope"].description,
)
@click.option(
    "--batch-size",
    "batch_size",
    type=_int_range(_EMB_DOC["batch_size"]),
    default=32,
    show_default=True,
    help=_EMB["batch_size"].description,
)
@click.option(
    "--every-n-frames",
    "every_n_frames",
    default=None,
    type=int,
    help=_EMB["every_n_frames"].description,
)
@_ALIGN_OPTS
@click.option(
    "--keypoint-ref",
    "keypoint_ref",
    default=None,
    help="RECORDING:kt_NN — derive the keypoint track kind from this run and resolve each "
    "recording's own match (alternative to --keypoint-run, for divergent ids).",
)
@_BACKEND_OPT
@_DEVICE_OPT
def process_embedding(
    experiment_dir: Path,
    recording_ids: tuple[str, ...],
    tracking_run_id: str | None,
    tracking_ref: str | None,
    track_ids: str | None,
    all_tracks: bool,
    model_id: str,
    model_variant: str | None,
    input_size: int | None,
    readout: str | None,
    renderer_size: int | None,
    constructor: str | None,
    canvas_scope: str,
    batch_size: int,
    every_n_frames: int | None,
    align: str,
    keypoint_run_id: str | None,
    heading: str | None,
    max_gap_s: float | None,
    mirror_hold_s: float | None,
    facing_camera_ratio: float | None,
    keypoint_ref: str | None,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Embed across the selected recordings; writes an experiment run (run_NNN).

    Point at the tracks with either ``--tracking track_NN`` (that id must be complete
    in every selected recording) or ``--tracking-ref RECORDING:track_NN`` (derive the
    tracking kind from one run and resolve each recording's own match). ``--resize`` is
    omitted by default, so the renderer emits the embedding model's own input size.
    ``--align rotate|mirror`` turns each animal to face right, reading the heading
    from ``--keypoint-run kt_NN`` (or ``--keypoint-ref RECORDING:kt_NN``) through the saved
    heading definition ``--heading NAME`` (§14)."""
    if bool(tracking_run_id) == bool(tracking_ref):
        raise click.ClickException("pass exactly one of --tracking or --tracking-ref")
    if keypoint_run_id and keypoint_ref:
        raise click.ClickException("pass one of --keypoint-run or --keypoint-ref")
    if all_tracks and track_ids:
        raise click.ClickException("use either --tracks or --all-tracks, not both")
    if not all_tracks and not track_ids:
        raise click.ClickException("provide --tracks or --all-tracks")
    ids: list[int] = []
    if not all_tracks:
        try:
            ids = [int(t.strip()) for t in track_ids.split(",") if t.strip()]
        except ValueError:
            raise click.ClickException(
                f"--tracks must be comma-separated ints, got {track_ids!r}"
            )
    proj = _open_experiment(experiment_dir)
    config = {
        "model_id": model_id,
        "model_variant": model_variant,
        "readout": readout,
        "input_size": input_size,
        "constructor": constructor or DEFAULT_CONSTRUCTOR,
        "resize": renderer_size,
        "canvas_scope": canvas_scope,
        "batch_size": batch_size,
        "every_n_frames": every_n_frames,
        "all_tracks": all_tracks,
        "track_ids": ids,
        "compute": _compute_target(backend, devices),
        **_alignment_config(
            align, heading, max_gap_s, mirror_hold_s, facing_camera_ratio
        ),
    }
    if tracking_ref:
        config["tracking_spec"] = _spec_from_ref(proj, "tracking", tracking_ref)
    else:
        config["tracking_run_id"] = tracking_run_id
    if keypoint_ref:
        config["keypoint_track_spec"] = _spec_from_ref(
            proj, "keypoint_track", keypoint_ref
        )
    else:
        config["keypoint_run_id"] = keypoint_run_id
    try:
        refs = _resolve_recording_refs(proj, recording_ids)
        run = _with_stage_progress(
            lambda bar: proj.run_processing("embedding", refs, config, progress=bar)
        )
    except (ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted processing run: {run.id}")
    _stream_processing_progress(run, "embedding")
    _check_status(run, "processing")


@process.command("list")
@_EXPERIMENT_OPT
def process_list(experiment_dir: Path) -> None:
    """List the experiment's processing runs (run_NNN)."""
    proj = _open_experiment(experiment_dir)
    runs = proj.processing_runs
    if not runs:
        click.echo("(no processing runs)")
        return
    for r in runs:
        n = len(r.get("recording_ids", []))
        click.echo(
            f"  {str(r.get('id')):10s} module={str(r.get('module_id')):10s} "
            f"status={str(r.get('status')):10s} recordings={n}"
        )


@process.command("status")
@click.argument("run_id")
@_EXPERIMENT_OPT
def process_status(run_id: str, experiment_dir: Path) -> None:
    """Show one processing run: module, status, and per-recording run ids."""
    proj = _open_experiment(experiment_dir)
    record = _processing_record(proj, run_id, experiment_dir)
    click.echo(f"run: {record.get('id')}")
    click.echo(f"  module: {record.get('module_id')}")
    click.echo(f"  status: {record.get('status')}")
    for sid, info in (record.get("per_recording") or {}).items():
        click.echo(f"  {sid}: run={info.get('run_id')} status={info.get('status')}")


@process.command("attach")
@click.argument("run_id")
@_EXPERIMENT_OPT
def process_attach(run_id: str, experiment_dir: Path) -> None:
    """Reattach to a processing run — stream a live one's progress (it keeps running
    with the app/CLI closed) or print a finished one's status. Ctrl-C detaches; it
    does not cancel the run (use ``process cancel`` for that)."""
    proj = _open_experiment(experiment_dir)
    record = _processing_record(proj, run_id, experiment_dir)
    handle = proj.attach_run(run_id)
    _stream_processing_progress(handle, record.get("module_id") or "run")
    click.echo(f"status: {handle.status}")


@process.command("retry")
@click.argument("run_id")
@_EXPERIMENT_OPT
def process_retry(run_id: str, experiment_dir: Path) -> None:
    """Re-run a finished run's incomplete recordings, into the same run record.

    Every recording that isn't complete — failed, cancelled, or never launched — is run
    again under a fresh per-recording id; the completed ones are left alone. The run rolls
    up to complete once all of them are. Refuses if any incomplete recording no longer
    validates, naming each one.
    """
    proj = _open_experiment(experiment_dir)
    record = _processing_record(proj, run_id, experiment_dir)
    try:
        handle = _with_stage_progress(lambda bar: proj.retry_run(run_id, progress=bar))
    except (ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    click.echo(f"retrying processing run: {run_id}")
    _stream_processing_progress(handle, record.get("module_id") or "run")
    _check_status(handle, "processing")


@process.command("cancel")
@click.argument("run_id")
@_EXPERIMENT_OPT
def process_cancel(run_id: str, experiment_dir: Path) -> None:
    """Cancel a running processing run — stops its coordinator and in-flight recordings.

    Recordings already finished keep their results, and recordings not yet reached are
    listed as cancelled; `process retry` picks up the rest.
    """
    proj = _open_experiment(experiment_dir)
    _processing_record(proj, run_id, experiment_dir)
    try:
        proj.cancel_run(run_id)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    click.echo(f"cancelled processing run: {run_id}")


def _processing_record(proj: Experiment, run_id: str, experiment_dir: Path) -> dict:
    """The run record, or an actionable ClickException naming the experiment."""
    record = next((r for r in proj.processing_runs if r.get("id") == run_id), None)
    if record is None:
        raise click.ClickException(f"no processing run {run_id!r} in {experiment_dir}")
    return record


# --------------------------------------------------------------------------- #
# dataset (training datasets, §12)
# --------------------------------------------------------------------------- #
@main.group()
def dataset() -> None:
    """Training datasets (§12) — export YOLO images+labels from a processing run."""


@dataset.command("create")
@_EXPERIMENT_OPT
@click.option("--name", required=True, type=str, help="Dataset name.")
@click.option(
    "--source-run",
    "source_run_id",
    required=True,
    type=str,
    help="Experiment processing run to export (run_NNN; detection or tracking).",
)
@click.option(
    "--every-n",
    "every_n",
    type=click.IntRange(min=1),
    default=None,
    help="Keep every Nth frame (default: all frames).",
)
@click.option(
    "--val-split",
    "val_split",
    type=FINITE_FLOAT,
    default=0.15,
    show_default=True,
    help="Fraction of frames held out for validation.",
)
@click.option(
    "--min-contour-area",
    "min_contour_area",
    type=int,
    default=100,
    show_default=True,
    help="Drop polygons smaller than this area (px).",
)
def dataset_create(
    experiment_dir: Path,
    name: str,
    source_run_id: str,
    every_n: int | None,
    val_split: float,
    min_contour_area: int,
) -> None:
    """Export a YOLO dataset from a processing run; blocks until complete."""
    proj = _open_experiment(experiment_dir)
    frame_selection = {"mode": "every_n", "n": every_n} if every_n else {"mode": "all"}
    try:
        run = proj.generate_dataset(
            name=name,
            source_run_id=source_run_id,
            frame_selection=frame_selection,
            val_split=val_split,
            min_contour_area=min_contour_area,
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted dataset: {run.id}")
    _stream_progress(run, "dataset")
    _check_status(run, "dataset")


@dataset.command("label")
@_EXPERIMENT_OPT
@click.option("--name", required=True, type=str, help="Dataset name.")
@click.option(
    "--recording",
    "recording_ids",
    multiple=True,
    type=str,
    help=(
        "Source recording id (repeatable; default: every recording in the experiment "
        "that has video)."
    ),
)
@click.option(
    "--class",
    "class_specs",
    multiple=True,
    metavar="NAME:PROMPT",
    help="A class as NAME:PROMPT, e.g. --class 'mouse:a mouse' (repeatable).",
)
@click.option(
    "--model",
    "model_id",
    default="sam3",
    show_default=True,
    type=str,
    help=_LAB["model_id"].description,
)
@click.option(
    "--model-variant",
    "model_variant",
    default=None,
    type=str,
    help=_LAB["model_variant"].description,
)
@click.option(
    "--n-frames",
    "n_frames",
    required=True,
    type=click.IntRange(min=1),
    help=_LAB["n_frames"].description,
)
@click.option(
    "--sampling",
    "sampling",
    default="stratified",
    show_default=True,
    type=click.Choice(list(get_args(_LAB["sampling"].annotation))),
    help=_LAB["sampling"].description,
)
@click.option(
    "--logit-threshold",
    "logit_threshold",
    default=0.5,
    show_default=True,
    type=FINITE_FLOAT,
    help=_LAB["logit_threshold"].description,
)
@click.option(
    "--min-contour-area",
    "min_contour_area",
    default=100,
    show_default=True,
    type=int,
    help=_LAB["min_contour_area"].description,
)
@click.option(
    "--val-split",
    "val_split",
    default=0.15,
    show_default=True,
    type=FINITE_FLOAT,
    help=_LAB["val_split"].description,
)
@click.option(
    "--background-share",
    "background_share",
    default=_LAB["background_share"].default,
    show_default=True,
    type=finite_float_range(0.0, 0.5),
    help=_LAB["background_share"].description,
)
@_BACKEND_OPT
@_DEVICE_OPT
def dataset_label(
    experiment_dir: Path,
    name: str,
    recording_ids: tuple[str, ...],
    class_specs: tuple[str, ...],
    model_id: str,
    model_variant: str | None,
    n_frames: int,
    sampling: str,
    logit_threshold: float,
    min_contour_area: int,
    val_split: float,
    background_share: float,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Label a random frame sample with a prompt-capable model into a YOLO dataset; blocks until
    complete."""
    proj = _open_experiment(experiment_dir)
    if not class_specs:
        raise click.ClickException("provide at least one --class NAME:PROMPT")
    classes = []
    for i, spec in enumerate(class_specs):
        cname, sep, prompt = spec.partition(":")
        if not sep or not cname.strip() or not prompt.strip():
            raise click.ClickException(f"--class must be NAME:PROMPT, got {spec!r}")
        classes.append(
            LabelingClass(class_id=i, name=cname.strip(), prompt=prompt.strip())
        )
    try:
        run = proj.label_dataset(
            name=name,
            classes=classes,
            n_frames=n_frames,
            source_recording_ids=list(recording_ids) or None,
            logit_threshold=logit_threshold,
            min_contour_area=min_contour_area,
            val_split=val_split,
            background_share=background_share,
            compute=_compute_target(backend, devices),
            sampling=sampling,
            model_id=model_id,
            model_variant=model_variant,
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted labeling: {run.id}")
    _stream_progress(run, "labeling")
    _check_status(run, "labeling")


@dataset.command("list")
@_EXPERIMENT_OPT
def dataset_list(experiment_dir: Path) -> None:
    """List the experiment's training datasets."""
    proj = _open_experiment(experiment_dir)
    datasets = proj.datasets
    if not datasets:
        click.echo("(no datasets)")
        return
    for d in datasets:
        click.echo(
            f"  {d.id:10s} {d.name:20s} status={d.status:10s} "
            f"train={d.n_train} val={d.n_val}"
        )


@dataset.command("status")
@click.argument("dataset_id")
@_EXPERIMENT_OPT
def dataset_status(dataset_id: str, experiment_dir: Path) -> None:
    """Show one dataset's record."""
    proj = _open_experiment(experiment_dir)
    try:
        record = proj.dataset_record(dataset_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    for key in ("id", "name", "status", "n_train", "n_val"):
        if key in record:
            click.echo(f"  {key}: {record[key]}")
    classes = record.get("classes")
    if classes:
        click.echo(f"  classes: {', '.join(c['name'] for c in classes)}")


@dataset.command("attach")
@click.argument("dataset_id")
@_EXPERIMENT_OPT
def dataset_attach(dataset_id: str, experiment_dir: Path) -> None:
    """Reattach to a dataset export — stream a live one's progress (it keeps running with the
    app/CLI closed) or print a finished one's status. Ctrl-C detaches; it does not cancel
    the run (use ``dataset cancel`` for that)."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.dataset_record(dataset_id)  # ClickException rather than a traceback
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    handle = proj.attach_dataset_run(dataset_id)
    _stream_progress(handle, "dataset")
    click.echo(f"status: {handle.status}")


@dataset.command("delete")
@click.argument("dataset_id")
@_EXPERIMENT_OPT
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def dataset_delete(dataset_id: str, experiment_dir: Path, yes: bool) -> None:
    """Delete a training dataset (images + labels + dataset.yaml)."""
    proj = _open_experiment(experiment_dir)
    if not yes:
        click.confirm(f"delete dataset {dataset_id!r}?", abort=True)
    try:
        proj.delete_dataset(dataset_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted dataset: {dataset_id}")


# --------------------------------------------------------------------------- #
# train (custom-model training, §12)
# --------------------------------------------------------------------------- #
@main.group()
def train() -> None:
    """Custom detection-model training (§12) — fit a model from a dataset."""


@train.command("run")
@_EXPERIMENT_OPT
@click.option("--name", required=True, type=str, help="Trained-model name.")
@click.option(
    "--dataset", "dataset_id", required=True, type=str, help="Dataset id (dataset_NNN)."
)
@click.option(
    "--trainer",
    "trainer_id",
    default="ultralytics_yolo",
    show_default=True,
    type=str,
    help="Trainer id.",
)
@click.option(
    "--base-model",
    "base_model",
    default="yolo11s-seg.pt",
    show_default=True,
    type=str,
    help="Pretrained starting weights — a YOLO-seg checkpoint (labels are polygons).",
)
@click.option("--epochs", type=int, default=None, help=_TRAIN["epochs"].description)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=None,
    help=_TRAIN["batch_size"].description,
)
@click.option("--imgsz", type=int, default=None, help=_TRAIN["imgsz"].description)
@click.option("--patience", type=int, default=None, help=_TRAIN["patience"].description)
@click.option(
    "--hyperparam",
    "raw_hyperparams",
    multiple=True,
    metavar="KEY=VALUE",
    help="Set any trainer hyperparam (repeatable); coerced to its declared type.",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default=None,
    help="Compute backend (default: the best available: CUDA, then MPS, then CPU).",
)
@_DEVICE_OPT
def train_run(
    experiment_dir: Path,
    name: str,
    dataset_id: str,
    trainer_id: str,
    base_model: str,
    epochs: int | None,
    batch_size: int | None,
    imgsz: int | None,
    patience: int | None,
    raw_hyperparams: tuple[str, ...],
    backend: str | None,
    devices: tuple[int, ...],
) -> None:
    """Train a detection model from a dataset; blocks until complete."""
    proj = _open_experiment(experiment_dir)
    hyperparams: dict = {}
    if epochs is not None:
        hyperparams["epochs"] = epochs
    if batch_size is not None:
        hyperparams["batch_size"] = batch_size
    if imgsz is not None:
        hyperparams["imgsz"] = imgsz
    if patience is not None:
        hyperparams["patience"] = patience
    if raw_hyperparams:
        import detail.trainers.ultralytics_yolo  # noqa: F401 — registers TRAINERS

        from detail.core.registries import TRAINERS

        trainer_cls = TRAINERS.get(trainer_id)
        if trainer_cls is None:
            raise click.ClickException(
                f"unknown trainer {trainer_id!r}; available: {sorted(TRAINERS)}"
            )
        model_cls = getattr(trainer_cls, "config_model", None)
        schema = (
            {
                name: (field.annotation, field.default)
                for name, field in model_cls.model_fields.items()
            }
            if model_cls is not None
            else {}
        )
        for item in raw_hyperparams:
            if "=" not in item:
                raise click.ClickException(
                    f"--hyperparam must be KEY=VALUE, got {item!r}"
                )
            key, raw = item.split("=", 1)
            key = key.strip()
            if key in hyperparams:
                raise click.ClickException(
                    f"hyperparam {key!r} set twice (a dedicated flag and --hyperparam)"
                )
            hyperparams[key] = _coerce_hyperparam(schema, key, raw)
    try:
        run = proj.train_model(
            name=name,
            dataset_id=dataset_id,
            trainer_id=trainer_id,
            base_model=base_model,
            hyperparams=hyperparams,
            compute=_compute_target(backend, devices) if backend else None,
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted training: {run.id}")
    _stream_training_progress(run, "train")
    _check_status(run, "training")


@train.command("resume")
@click.argument("model_id")
@_EXPERIMENT_OPT
def train_resume(model_id: str, experiment_dir: Path) -> None:
    """Resume a cancelled or crashed training from its last checkpoint (not from epoch 0).

    A training stopped with `train stop` is finished, not interrupted: it keeps its best
    weights and has nothing to resume.
    """
    proj = _open_experiment(experiment_dir)
    try:
        run = proj.resume_training(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"resumed training: {run.id}")
    _stream_training_progress(run, "train")
    _check_status(run, "training")


def _best_epoch_line(history: list[dict], record: dict) -> str:
    """The best mAP50 so far and the epoch it was reached, for `train status`.

    The CLI half of `gui/training_curve.py::best_line` and deliberately not shared with it: that
    one composes a caption under a plot the reader can already see the shape of, this one is the
    only thing a terminal gets, and the two are printed in different sentences. What they share
    is the rule — name the epoch as well as the score, because a number alone cannot be compared
    with anything; what makes 0.847 actionable is that it was reached at epoch 31 and the run is
    on epoch 58.
    """
    scored = [
        (int(row["epoch"]), float(row["box_map50"]))
        for row in history
        if row.get("box_map50") is not None and row.get("epoch") is not None
    ]
    if not scored:
        return "best so far: (no scored epoch yet)"
    epoch, score = max(scored, key=lambda pair: pair[1])
    latest = max(e for e, _ in scored)
    total = (record.get("hyperparams") or {}).get("epochs")
    line = f"best so far: {score:.3f} mAP50 at epoch {epoch}"
    if latest > epoch:
        behind = latest - epoch
        line += f" (no improvement in {behind} epoch{'s' if behind != 1 else ''})"
    if total:
        line += f", {latest} of {total} epochs done"
    return line


@train.command("stop")
@click.argument("model_id")
@_EXPERIMENT_OPT
def train_stop(model_id: str, experiment_dir: Path) -> None:
    """Stop a training run at its next epoch, keeping the best weights so far.

    The counterpart to interrupting `train run`, which kills the worker: that stops it now and
    the model is lost, because the block that records the weights never runs. This asks instead,
    so the run finishes the epoch it is on and writes its own record. It is not instant — an
    epoch on real data is minutes.
    """
    proj = _open_experiment(experiment_dir)
    try:
        proj.stop_model(model_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    click.echo(
        f"asked {model_id} to stop after the current epoch; "
        "its best weights will be saved. Watch it with: detail train attach " + model_id
    )


@train.command("list")
@_EXPERIMENT_OPT
def train_list(experiment_dir: Path) -> None:
    """List the experiment's trained models."""
    proj = _open_experiment(experiment_dir)
    models = proj.models
    if not models:
        click.echo("(no trained models)")
        return
    for m in models:
        click.echo(
            f"  {m.id:10s} {m.name:20s} status={m.status:10s} "
            f"trainer={m.trainer_id} dataset={m.dataset_id}"
        )


@train.command("status")
@click.argument("model_id")
@_EXPERIMENT_OPT
def train_status(model_id: str, experiment_dir: Path) -> None:
    """Show one trained model's record (classes, final metrics, best epoch so far)."""
    proj = _open_experiment(experiment_dir)
    try:
        record = proj.model_record(model_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    for key in ("id", "name", "status", "trainer_id", "dataset_id", "base_model"):
        if key in record:
            click.echo(f"  {key}: {record[key]}")
    from detail.api import models as api_models

    trained_with = api_models.training_summary(record)
    if trained_with:
        click.echo(f"  trained with: {trained_with}")
    if record.get("classes"):
        click.echo(f"  classes: {record['classes']}")
    if record.get("metrics"):
        click.echo(f"  metrics: {record['metrics']}")
    # The line `train stop` is decided from. Without it the CLI can stop a training and gives
    # nobody anything to stop it *on* — the GUI grew a curve for exactly this question, and a
    # terminal cannot draw one, so it gets the sentence the curve is captioned with.
    history = proj.model_metric_history(model_id)
    if history:
        click.echo(f"  {_best_epoch_line(history, record)}")


@train.command("attach")
@click.argument("model_id")
@_EXPERIMENT_OPT
def train_attach(model_id: str, experiment_dir: Path) -> None:
    """Reattach to a training run — stream a live one's progress (it keeps running with the
    app/CLI closed) or print a finished one's status. Ctrl-C detaches; it does not cancel
    the run (use ``train cancel`` for that)."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.model_record(model_id)  # ClickException rather than a traceback
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    handle = proj.attach_model_run(model_id)
    _stream_progress(handle, "train")
    click.echo(f"status: {handle.status}")


@train.command("delete")
@click.argument("model_id")
@_EXPERIMENT_OPT
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def train_delete(model_id: str, experiment_dir: Path, yes: bool) -> None:
    """Delete a trained model (model.yaml + trainer run + weights)."""
    proj = _open_experiment(experiment_dir)
    if not yes:
        click.confirm(f"delete model {model_id!r}?", abort=True)
    try:
        proj.delete_model(model_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted model: {model_id}")


@train.command("promote")
@click.argument("model_id")
@_EXPERIMENT_OPT
@click.option(
    "--name",
    default=None,
    type=str,
    help="Name for the promoted global model (default: the trained model's name).",
)
def train_promote(model_id: str, experiment_dir: Path, name: str | None) -> None:
    """Promote a trained model into the machine-global catalog — usable in any experiment.

    Copies the model (weights + classes) into the global catalog with its provenance (this
    experiment + the source dataset + base model + metrics); the original stays put.
    """
    proj = _open_experiment(experiment_dir)
    try:
        gid = proj.promote_model(model_id, name=name)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"promoted {model_id} → global model: {gid}")


# --------------------------------------------------------------------------- #
# distill (label → train, one durable run — §12)
# --------------------------------------------------------------------------- #
@main.group()
def distill() -> None:
    """Distill a fast detector — label with a prompt-capable model, then train YOLO (§12)."""


@distill.command("run")
@_EXPERIMENT_OPT
@click.option(
    "--name",
    required=True,
    type=str,
    help="Distill run name (seeds the dataset + model).",
)
@click.option(
    "--recording",
    "recording_ids",
    multiple=True,
    type=str,
    help=(
        "Source recording id (repeatable; default: every recording in the experiment "
        "that has video)."
    ),
)
@click.option(
    "--class",
    "class_specs",
    multiple=True,
    metavar="NAME:PROMPT",
    help="A class as NAME:PROMPT, e.g. --class 'mouse:a mouse' (repeatable).",
)
@click.option(
    "--n-frames",
    "n_frames",
    required=True,
    type=click.IntRange(min=1),
    help=_LAB["n_frames"].description,
)
@click.option(
    "--sampling",
    default="stratified",
    show_default=True,
    type=click.Choice(list(get_args(_LAB["sampling"].annotation))),
    help=_LAB["sampling"].description,
)
@click.option(
    "--logit-threshold",
    "logit_threshold",
    default=0.5,
    show_default=True,
    type=FINITE_FLOAT,
    help=_LAB["logit_threshold"].description,
)
@click.option(
    "--min-contour-area",
    "min_contour_area",
    default=100,
    show_default=True,
    type=int,
    help=_LAB["min_contour_area"].description,
)
@click.option(
    "--val-split",
    "val_split",
    default=0.15,
    show_default=True,
    type=FINITE_FLOAT,
    help=_LAB["val_split"].description,
)
@click.option(
    "--background-share",
    "background_share",
    default=_LAB["background_share"].default,
    show_default=True,
    type=finite_float_range(0.0, 0.5),
    help=_LAB["background_share"].description,
)
@click.option(
    "--label-model",
    "label_model_id",
    default="sam3",
    show_default=True,
    type=str,
    help=_LAB["model_id"].description,
)
@click.option(
    "--label-model-variant",
    "label_model_variant",
    default=None,
    type=str,
    help=_LAB["model_variant"].description,
)
@click.option(
    "--base-model",
    "base_model",
    default="yolo11s-seg.pt",
    show_default=True,
    type=str,
    help="Pretrained YOLO-seg weights the trainer fine-tunes.",
)
@click.option(
    "--trainer",
    "trainer_id",
    default="ultralytics_yolo",
    show_default=True,
    type=str,
    help="Trainer id.",
)
@click.option("--epochs", type=int, default=None, help=_TRAIN["epochs"].description)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=None,
    help=_TRAIN["batch_size"].description,
)
@click.option("--imgsz", type=int, default=None, help=_TRAIN["imgsz"].description)
@click.option("--patience", type=int, default=None, help=_TRAIN["patience"].description)
@click.option(
    "--label-backend",
    "label_backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default=None,
    help="Labeling backend (SAM3 requires CUDA; default: best available).",
)
@click.option(
    "--label-device",
    "label_devices",
    multiple=True,
    type=int,
    help="Labeling CUDA device index (repeatable).",
)
@click.option(
    "--train-backend",
    "train_backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default=None,
    help="Training backend (default: the best available: CUDA, then MPS, then CPU).",
)
@click.option(
    "--train-device",
    "train_devices",
    multiple=True,
    type=int,
    help="Training CUDA device index (repeatable).",
)
def distill_run(
    experiment_dir: Path,
    name: str,
    recording_ids: tuple[str, ...],
    class_specs: tuple[str, ...],
    n_frames: int,
    sampling: str,
    logit_threshold: float,
    min_contour_area: int,
    val_split: float,
    background_share: float,
    label_model_id: str,
    label_model_variant: str | None,
    base_model: str,
    trainer_id: str,
    epochs: int | None,
    batch_size: int | None,
    imgsz: int | None,
    patience: int | None,
    label_backend: str | None,
    label_devices: tuple[int, ...],
    train_backend: str | None,
    train_devices: tuple[int, ...],
) -> None:
    """Label a frame sample, then train a YOLO detector on it — one durable run; blocks
    until complete (Ctrl-C detaches, the run keeps going in the background)."""
    proj = _open_experiment(experiment_dir)
    if not class_specs:
        raise click.ClickException("provide at least one --class NAME:PROMPT")
    classes = []
    for i, spec in enumerate(class_specs):
        cname, sep, prompt = spec.partition(":")
        if not sep or not cname.strip() or not prompt.strip():
            raise click.ClickException(f"--class must be NAME:PROMPT, got {spec!r}")
        classes.append(
            LabelingClass(class_id=i, name=cname.strip(), prompt=prompt.strip())
        )
    hyperparams: dict = {}
    if epochs is not None:
        hyperparams["epochs"] = epochs
    if batch_size is not None:
        hyperparams["batch_size"] = batch_size
    if imgsz is not None:
        hyperparams["imgsz"] = imgsz
    if patience is not None:
        hyperparams["patience"] = patience
    try:
        run = proj.distill(
            name=name,
            classes=classes,
            n_frames=n_frames,
            source_recording_ids=list(recording_ids) or None,
            sampling=sampling,
            logit_threshold=logit_threshold,
            min_contour_area=min_contour_area,
            val_split=val_split,
            background_share=background_share,
            label_model_id=label_model_id,
            label_model_variant=label_model_variant,
            label_compute=(
                _compute_target(label_backend, label_devices) if label_backend else None
            ),
            base_model=base_model,
            trainer_id=trainer_id,
            hyperparams=hyperparams or None,
            train_compute=(
                _compute_target(train_backend, train_devices) if train_backend else None
            ),
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted distill: {run.id}")
    _stream_distill_progress(run, "distill")
    _check_status(run, "distill")


@distill.command("list")
@_EXPERIMENT_OPT
def distill_list(experiment_dir: Path) -> None:
    """List the experiment's distill runs (distill_NNN)."""
    proj = _open_experiment(experiment_dir)
    runs = proj.distill_runs
    if not runs:
        click.echo("(no distill runs)")
        return
    for r in runs:
        stages = r.get("stages") or {}
        click.echo(
            f"  {str(r.get('id')):12s} {str(r.get('name')):20s} "
            f"status={str(r.get('status')):10s} "
            f"label={(stages.get('label') or {}).get('status')} "
            f"train={(stages.get('train') or {}).get('status')}"
        )


@distill.command("status")
@click.argument("run_id")
@_EXPERIMENT_OPT
def distill_status(run_id: str, experiment_dir: Path) -> None:
    """Show one distill run: status + each stage's produced dataset / model id."""
    proj = _open_experiment(experiment_dir)
    try:
        record = proj.distill_run_record(run_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    click.echo(f"distill: {record.get('id')} ({record.get('name')})")
    click.echo(f"  kind: {record.get('kind') or 'detection'}")
    click.echo(f"  status: {record.get('status')}")
    stages = record.get("stages") or {}
    label = stages.get("label") or {}
    train = stages.get("train") or {}
    click.echo(
        f"  label: status={label.get('status')} dataset={label.get('dataset_id')}"
    )
    click.echo(f"  train: status={train.get('status')} model={train.get('model_id')}")
    if "register" in stages:
        # A keypoint distill's third stage (§12 *Distill keypoints*).
        register = stages.get("register") or {}
        click.echo(f"  register: status={register.get('status')} role=keypoints")
    if record.get("error"):
        click.echo(f"  error: {record['error']}")


@distill.command("attach")
@click.argument("run_id")
@_EXPERIMENT_OPT
def distill_attach(run_id: str, experiment_dir: Path) -> None:
    """Reattach to a distill run — stream a live one (it keeps running with the CLI closed)
    or print a finished one's status. Ctrl-C detaches; it does not cancel the run (use
    ``distill cancel`` for that)."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.distill_run_record(run_id)  # FileNotFoundError → actionable
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    handle = proj.attach_distill_run(run_id)
    _stream_distill_progress(handle, "distill")
    click.echo(f"status: {handle.status}")


@distill.command("resume")
@click.argument("run_id")
@_EXPERIMENT_OPT
def distill_resume(run_id: str, experiment_dir: Path) -> None:
    """Resume a failed / cancelled distill run — reuse a complete dataset, resume a partial
    model — into the same record."""
    proj = _open_experiment(experiment_dir)
    try:
        run = proj.resume_distill(run_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"resumed distill: {run.id}")
    _stream_distill_progress(run, "distill")
    _check_status(run, "distill")


@distill.command("cancel")
@click.argument("run_id")
@_EXPERIMENT_OPT
def distill_cancel(run_id: str, experiment_dir: Path) -> None:
    """Cancel a running distill run — stops its coordinator and the live stage worker."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.cancel_distill(run_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"cancelled distill run: {run_id}")


@distill.command("delete")
@click.argument("run_id")
@_EXPERIMENT_OPT
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def distill_delete(run_id: str, experiment_dir: Path, yes: bool) -> None:
    """Delete a distill run's index record. The produced dataset and trained model are
    independent and are left intact (delete them with ``dataset delete`` / ``train delete``).
    """
    proj = _open_experiment(experiment_dir)
    if not yes:
        click.confirm(
            f"delete distill run {run_id!r}? (its dataset + model stay)", abort=True
        )
    try:
        proj.delete_distill(run_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted distill run: {run_id}")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@main.group()
def model() -> None:
    """Model readiness and install (§5)."""


@model.command("check")
@click.argument("model_id")
@click.option(
    "--variant",
    default=None,
    help="Model size/variant to check (default: the model's default variant). Each "
    "variant is a separate download.",
)
@click.option(
    "--experiment",
    "experiment_dir",
    type=UserPath(file_okay=False, path_type=Path),
    help="Accepted for older scripts and ignored: models are machine-global, so no "
    "experiment is needed.",
)
def model_check(
    model_id: str, variant: str | None, experiment_dir: Path | None
) -> None:
    """Print readiness reason + action URL for a model."""
    from detail.api import models as api_models

    try:
        r = api_models.model_readiness(model_id, variant)
    except ValueError as e:
        raise click.ClickException(str(e))
    label = f"{model_id}/{variant}" if variant else model_id
    click.echo(f"{label}: ready={r.ready} reason={r.reason}")
    if r.message:
        click.echo(f"  message: {r.message}")
    if r.action_url:
        click.echo(f"  action : {r.action_url}")


@model.command("install")
@click.argument("model_id")
@click.option(
    "--variant",
    default=None,
    help="Model size/variant to download (default: the model's default variant). Each "
    "variant is a separate download — installing one does not install the others.",
)
@click.option(
    "--experiment",
    "experiment_dir",
    type=UserPath(file_okay=False, path_type=Path),
    help="Accepted for older scripts and ignored: the weights cache is machine-global, so "
    "no experiment is needed.",
)
def model_install(
    model_id: str, variant: str | None, experiment_dir: Path | None
) -> None:
    """Download a model's weights into the global cache; idempotent.

    A run gates on the exact variant it is configured with, so install the size you
    plan to run — `--variant convnext_tiny` is not covered by installing the default.
    """
    from detail.api import models as api_models

    last_file: dict[str, str | None] = {"name": None}

    def on_progress(p: InstallProgress) -> None:
        if p.current_file and p.current_file != last_file["name"]:
            last_file["name"] = p.current_file
            click.echo(f"  {p.current_file}", err=True)
        if p.bytes_total:
            pct = 100.0 * p.bytes_done / p.bytes_total
            click.echo(
                f"    {p.bytes_done / 1e6:.1f}/{p.bytes_total / 1e6:.1f} MB ({pct:.1f}%)",
                err=True,
            )

    try:
        api_models.install_model(model_id, on_progress, variant=variant)
    except (ValueError, ModelNotReadyError) as e:
        raise click.ClickException(str(e))
    label = f"{model_id}/{variant}" if variant else model_id
    click.echo(f"{label}: install complete")


#: ``model import --skeleton``'s help: the one statement of what the option takes and when.
_IMPORT_SKELETON_HELP = (
    "The skeleton a pose (keypoint) checkpoint's points follow: a preset, a library id "
    "(sk_NN) or a skeleton YAML. Required for a pose checkpoint, refused for a detection "
    "one. It must have the model's number of points, and left/right pairs that match the "
    "model's flip_idx; a pair named the wrong way round cannot be caught."
)


@model.command("import")
@click.argument("weights", type=UserPath(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--family",
    default=None,
    help="Model family. Leave it out: the checkpoint's task decides (a pose checkpoint "
    "imports as yolo_pose); when given it must agree.",
)
@click.option("--skeleton", default=None, help=_IMPORT_SKELETON_HELP)
@click.option(
    "--name", default=None, help="Name for the model (default: the weights filename)."
)
@click.option("--made-by", default=None, help="Who made the model (a plain name).")
@click.option("--description", default=None, help="What the model is for.")
@click.option("--notes", default=None, help="Optional free-text notes.")
@click.option(
    "--preview",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="An image to store as the model's preview.",
)
def model_import(
    weights: Path,
    family: str | None,
    skeleton: str | None,
    name: str | None,
    made_by: str | None,
    description: str | None,
    notes: str | None,
    preview: Path | None,
) -> None:
    """Import a checkpoint into the machine-global catalog (usable in any experiment).
    A detection checkpoint becomes a detector, its classes read from the checkpoint; a pose
    checkpoint becomes a keypoint model and needs --skeleton."""
    from detail.api import models as api_models

    try:
        model_id = api_models.import_model(
            weights,
            family=family,
            skeleton=skeleton,
            name=name,
            made_by=made_by,
            description=description,
            notes=notes,
            preview=preview,
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"imported model: {model_id}")
    _echo_model_skeleton(model_id)


def _echo_model_skeleton(model_id: str) -> None:
    """For a keypoint model: the skeleton library entry its skeleton matched or was added to."""
    from detail.api import models as api_models

    entry = api_models.imported_model(model_id).skeleton_ref
    if entry:
        click.echo(f"skeleton: {entry.get('name')} ({entry.get('id')}) in the library")


@model.command("show")
@click.argument("model_id")
def model_show(model_id: str) -> None:
    """Show one of your models' metadata + provenance (the model detail, as text)."""
    from detail.api import models as api_models

    try:
        ref = api_models.imported_model(model_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    click.echo(f"id:          {ref.id}")
    click.echo(f"name:        {ref.name}")
    click.echo(f"source:      {ref.source}")
    click.echo(f"role:        {ref.role}")
    if ref.skeleton is not None:
        entry = ref.skeleton_ref or {}
        library = f"{entry.get('name')} ({entry.get('id')})" if entry else "—"
        pairs = ", ".join(f"{a}/{b}" for a, b in ref.skeleton.pairs) or "none"
        click.echo(f"skeleton:    {library}")
        click.echo(f"points:      {', '.join(ref.skeleton.names)}")
        click.echo(f"pairs:       {pairs}")
    click.echo(f"created_at:  {ref.created_at or '—'}")
    click.echo(f"made_by:     {ref.made_by or '—'}")
    click.echo(f"description: {ref.description or '—'}")
    click.echo(f"notes:       {ref.notes or '—'}")
    click.echo(f"classes:     {', '.join(ref.classes) or '—'}")
    click.echo(f"preview:     {ref.preview_path or '—'}")
    if ref.source_experiments:
        projects = ", ".join(
            s.get("name") or s.get("id") or "?" for s in ref.source_experiments
        )
        click.echo(f"from:        {projects}")
    if ref.provenance:
        click.echo(f"base_model:  {ref.provenance.get('base_model') or '—'}")
        click.echo(f"metrics:     {ref.provenance.get('metrics') or {}}")
        click.echo(f"trained:     {api_models.training_summary(ref.provenance) or '—'}")


@model.command("export")
@click.argument("model_id")
@click.argument("dest", type=UserPath(dir_okay=False, path_type=Path))
def model_export(model_id: str, dest: Path) -> None:
    """Export one of your models to a .zip (weights + metadata + preview) to share it."""
    from detail.api import models as api_models

    bar = _StageProgress()
    try:
        written = api_models.export_model(model_id, dest, progress=bar)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    click.echo(f"exported {model_id} -> {written}")


@model.command("set-meta")
@click.argument("model_id")
@click.option("--made-by", default=None, help="Who made the model.")
@click.option("--description", default=None, help="What the model is for.")
@click.option("--notes", default=None, help="Optional free-text notes.")
@click.option(
    "--preview",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="An image to set as the model's preview.",
)
def model_set_meta(
    model_id: str,
    made_by: str | None,
    description: str | None,
    notes: str | None,
    preview: Path | None,
) -> None:
    """Edit one of your models' metadata (made-by / description / notes / preview)."""
    from detail.api import models as api_models

    if made_by is None and description is None and notes is None and preview is None:
        raise UserInputError(
            "nothing to change; pass --made-by, --description, --notes or --preview"
        )
    try:
        api_models.update_model_metadata(
            model_id, made_by=made_by, description=description, notes=notes
        )
        if preview is not None:
            api_models.set_model_preview(model_id, preview)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"updated: {model_id}")


@model.command("restore")
@click.argument("archive", type=UserPath(exists=True, dir_okay=False, path_type=Path))
def model_restore(archive: Path) -> None:
    """Restore a model exported with `model export` (a .zip) into the catalog."""
    from detail.api import models as api_models

    try:
        model_id = api_models.import_model_archive(archive)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"restored model: {model_id}")
    _echo_model_skeleton(model_id)


def _surfaced_model_variants() -> list:
    """`model list`'s rows, less any model this build hides (§25).

    Filtered where the catalog is listed, never at the id lookup: `model install <id>` and a run
    that names a hidden model still work, and so does a saved run that used one.
    """
    from detail.api import models as api_models

    return [r for r in api_models.list_model_variants() if surfaces.is_surfaced(r.id)]


@model.command("list")
def model_list() -> None:
    """List every machine-global model — built-in and imported — with install status.

    Answers "what can I run right now?" with no network calls: a run will not
    download weights, so anything marked `not installed` needs `model install <id>`
    first. Use `model check <id>` to find out why one cannot be installed.

    One line per installable unit — a varianted model lists each size separately,
    because each is its own download and a run gates on the size it was configured
    with. Install one with `model install <id> --variant <variant>`.
    """
    from detail.api import models as api_models

    rows = [
        (f"{r.id}/{r.variant}" if r.variant else r.id, r)
        for r in _surfaced_model_variants()
    ]
    width = max((len(label) for label, _ in rows), default=8)
    for label, r in rows:
        installed = api_models.is_installed(r.id, r.variant)
        status = "installed" if installed else "not installed"
        bits = [
            f"role={' · '.join(r.capabilities) or r.role}",
            f"arch={r.architecture}",
            f"source={r.source}",
        ]
        if r.classes:
            bits.append(f"classes={', '.join(r.classes)}")
        if r.role == "keypoint" and r.source != "built-in":
            sk = api_models.imported_model(r.id).skeleton
            if sk is not None:
                bits.append(f"skeleton={len(sk)} points ({', '.join(sk.names)})")
        if r.gated:
            bits.append("gated")
        click.echo(f"  {label:{width}s}  {status:13s}  {'  '.join(bits)}")


@model.command("remove")
@click.argument("model_id")
def model_remove(model_id: str) -> None:
    """Remove a machine-global imported model, or a folder an import stopped part-way by an
    earlier version left in the catalog."""
    from detail.api import models as api_models

    try:
        api_models.remove_imported_model(model_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    click.echo(f"removed imported model: {model_id}")


# --------------------------------------------------------------------------- #
# locations (per-recording ROI/POI spatial-features export)
# --------------------------------------------------------------------------- #
@main.group()
def locations() -> None:
    """Location export — per-frame, per-track spatial features vs ROIs/POIs."""


def _location_options(command):
    """The location export's measurement options, on any command that launches one.

    Help, bounds and defaults are the config model's (``configs/location_export.py``); each
    flag defaults to "not given", so the model's default is what an unset option resolves to
    and the two commands that take these cannot disagree about it.
    """
    options = [
        click.option(
            "--sample-fps", "sample_fps", default=None, type=float,
            help=_LOC["sample_fps"].description,
        ),  # fmt: skip
        click.option(
            "--remove-thin-parts/--keep-thin-parts", "remove_thin_parts", default=None,
            help=_LOC["remove_thin_parts"].description,
        ),  # fmt: skip
        click.option(
            "--thin-part-scale", "thin_part_scale", default=None, type=float,
            help=_LOC["thin_part_scale"].description
            + f" Default: {_LOC['thin_part_scale'].default}.",
        ),  # fmt: skip
    ]
    for option in reversed(options):
        command = option(command)
    return command


def _given_location_options(**values) -> dict:
    """The location options the user typed, by config field name; unset ones left out."""
    return {name: value for name, value in values.items() if value is not None}


@locations.command("export")
@_EXPERIMENT_OPT
@_RECORDING_OPT
@click.option(
    "--tracking", "tracking_run_id", required=True, type=str,
    help="Completed tracking run id (e.g. track_01).",
)  # fmt: skip
@click.option(
    "--out", "out_dir", required=True, type=UserPath(file_okay=False),
    help=(
        "Directory to create the export folder in; each export makes a new one holding "
        "tracks/roi_relations/point_relations CSVs + locations.yaml."
    ),
)  # fmt: skip
@_location_options
def locations_export(
    experiment_dir: Path,
    recording_id: str,
    tracking_run_id: str,
    out_dir: str,
    sample_fps: float | None,
    remove_thin_parts: bool | None,
    thin_part_scale: float | None,
) -> None:
    """Write per-frame, per-track spatial features (position, body shape, and position
    relative to the recording's ROIs/POIs) for a completed tracking run, keyed by t_ns.
    """
    sess = _recording(_open_experiment(experiment_dir), recording_id)
    options = _given_location_options(
        sample_fps=sample_fps,
        remove_thin_parts=remove_thin_parts,
        thin_part_scale=thin_part_scale,
    )
    try:
        out = sess.export_locations(tracking_run_id, out_dir, **options)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"exported: {out}")


# --------------------------------------------------------------------------- #
# analysis (experiment-scoped analysis runs, §16)
# --------------------------------------------------------------------------- #
@main.group()
def analysis() -> None:
    """Experiment-level analysis runs (§16) — recorded, batchable runs over a recording
    selection, written to analysis/<run_id>/. (The synchronous per-recording
    ``locations export`` above stays the quick scripting path.)"""


@analysis.command("run")
@_EXPERIMENT_OPT
@_RECORDINGS_OPT
@click.option(
    "--module", "module_id", default="location_export", show_default=True,
    help="Analysis module to run.",
)  # fmt: skip
@click.option(
    "--tracking", "tracking_run_id", default=None, type=str,
    help=_LOC["tracking_run_id"].description,
)  # fmt: skip
@click.option(
    "--tracking-ref", "tracking_refs", multiple=True,
    help="RECORDING:track_NN — derive the tracking kind from this run and resolve each "
    "recording's own match (alternative to --tracking). Repeat it to cover recordings "
    "tracked through different chains: each reads its run of the one named setting it has.",
)  # fmt: skip
@_location_options
def analysis_run(
    experiment_dir: Path,
    recording_ids: tuple[str, ...],
    module_id: str,
    tracking_run_id: str | None,
    tracking_refs: tuple[str, ...],
    sample_fps: float | None,
    remove_thin_parts: bool | None,
    thin_part_scale: float | None,
) -> None:
    """Launch an analysis run over the selected recordings; writes ``an_NN``."""
    if bool(tracking_run_id) == bool(tracking_refs):
        raise click.ClickException("pass exactly one of --tracking or --tracking-ref")
    options = _given_location_options(
        sample_fps=sample_fps,
        remove_thin_parts=remove_thin_parts,
        thin_part_scale=thin_part_scale,
    )
    if options and module_id != "location_export":
        # Applied or refused, never dropped: another module would ignore them in silence.
        given = ", ".join(f"--{name.replace('_', '-')}" for name in options)
        raise click.ClickException(
            f"{given} belong to the location_export module, not to {module_id}"
        )
    proj = _open_experiment(experiment_dir)
    config = {"source_recording_ids": list(recording_ids), **options}
    if tracking_refs:
        config.update(_spec_selection(proj, "tracking", tracking_refs))
    else:
        config["tracking_run_id"] = tracking_run_id
    try:
        run = proj.run_analysis(module_id, config)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted analysis: {run.id}")
    _stream_progress(run, "analysis")
    _check_status(run, "analysis")


@analysis.command("list")
@_EXPERIMENT_OPT
def analysis_list(experiment_dir: Path) -> None:
    """List the experiment's analysis runs (an_NN)."""
    proj = _open_experiment(experiment_dir)
    # One job.json read per run, so a listing never shows a status the job disagrees with.
    proj.reconcile_analysis_runs()
    runs = proj.analysis_runs
    if not runs:
        click.echo("(no analysis runs)")
        return
    for r in runs:
        click.echo(
            f"  {r.id:8s} module={r.module_id:16s} status={r.status:10s} "
            f"recordings={len(r.recording_ids)}"
        )


def _echo_setting(name: str, value, indent: str, mark: str = "") -> None:
    """One line of a run's settings, or an indented block for a nested one.

    A run's parameters are not all scalars: a behavior scan carries its whole set of behaviors
    with every example clip's recording, track and span. Printed as a value that is a plain
    2,000-character line, which is not output anybody reads — so a nested dict becomes a block
    and a list of structures becomes its length. What a person needs from this command is what
    the run was set to, and how many behaviors it carried is that; the clips themselves are in
    the record for a reader that wants them.
    """
    if isinstance(value, dict):
        click.echo(f"{indent}{name}:{mark}")
        for key in sorted(value):
            _echo_setting(key, value[key], indent + "  ")
        return
    if isinstance(value, list):
        scalars = all(isinstance(v, (str, int, float, bool)) for v in value)
        rendered = ", ".join(str(v) for v in value)
        if value and scalars and len(rendered) <= 60:
            click.echo(f"{indent}{name}: {rendered}{mark}")
        else:
            click.echo(f"{indent}{name}: {len(value)} item(s){mark}")
        return
    # `None` is a real setting on the optional knobs ("work it out for me"), so it is shown
    # rather than skipped — but not as the word `None`, which reads as a missing value.
    shown = "(not set)" if value is None else value
    click.echo(f"{indent}{name}: {shown}{mark}")


@analysis.command("status")
@click.argument("run_id")
@_EXPERIMENT_OPT
def analysis_status(run_id: str, experiment_dir: Path) -> None:
    """Show one analysis run: module, status, recordings, settings, and output artifacts.

    The **settings** block is the CLI's half of the GUI's *Refine*: the parameters this run was
    launched with, ready to re-pass to `analysis run`. A parameter that names another artifact
    (an embedding run, a tracking run) is marked `[selection]` — it is scoped to this experiment as
    it stands, so re-passing one only means what you want if that artifact is still the right
    input. The GUI re-asks for those rather than carrying them, for the same reason.
    """
    from detail.configs.describe import describe_config, iter_config_models
    from detail.storage import analysis_runs as analysis_store

    proj = _open_experiment(experiment_dir)
    try:
        # Against the job envelope first: a runner that wrote an optimistic status and then
        # died leaves a record that reads healthy (see `core/dataclasses.py::reconciled_status`).
        proj.reconcile_analysis_run(run_id)
        record = proj.analysis_record(run_id)
    except FileNotFoundError:
        raise click.ClickException(f"no analysis run {run_id!r} in {experiment_dir}")
    click.echo(f"analysis: {record.get('id')}")
    click.echo(f"  module: {record.get('module_id')}")
    click.echo(f"  status: {record.get('status')}")
    if record.get(analysis_store.STALE_KEY) is True:
        click.echo(
            "  results: out of date — the last change was saved but its results were not "
            f"rebuilt; rebuild with 'detail analysis rebuild {run_id}'"
        )
    if record.get("error"):
        click.echo(f"  error: {record['error']}")
    click.echo(f"  recordings: {', '.join(record.get('recording_ids', []))}")
    params = record.get("params") or {}
    if params:
        model = dict(iter_config_models()).get(record.get("module_id") or "")
        scopes = (
            {f["name"]: f["scope"] for f in describe_config(model)} if model else {}
        )
        click.echo("  settings:")
        for name in sorted(params):
            _echo_setting(
                name,
                params[name],
                "    ",
                " [selection]" if scopes.get(name) == "selection" else "",
            )
    for art in record.get("artifacts", []):
        click.echo(f"  artifact: {art.get('path')} ({art.get('kind')})")


@analysis.command("snapshots")
@click.argument("run_id")
@_EXPERIMENT_OPT
@click.option(
    "--snapshot",
    "snapshot_id",
    default=None,
    help="Show one snapshot in full rather than listing them.",
)
def analysis_snapshots(
    run_id: str, experiment_dir: Path, snapshot_id: str | None
) -> None:
    """List an authored run's pinned artifacts, or show one (§21).

    A snapshot is frozen when a number leaves the run — an export, a comparison, a finalize — so a
    result that was cited stays reachable after the run is reopened and re-cut. **Which comparison
    used one is derived** by reading the comparisons back, never stored on the snapshot.
    """
    proj = _open_experiment(experiment_dir)
    try:
        record = proj.analysis_record(run_id)
    except FileNotFoundError:
        raise click.ClickException(f"no analysis run {run_id!r} in {experiment_dir}")
    run = proj.authored_run(run_id)
    if run is None:
        raise click.ClickException(
            f"{run_id} is a {record.get('module_id')!r} run, which is not authored and pins "
            "nothing. Snapshots belong to runs a person interprets (§21)."
        )
    used = proj.comparisons_using(run_id)

    if snapshot_id is not None:
        try:
            snap = run.snapshot(snapshot_id)
        except FileNotFoundError:
            have = ", ".join(s["snapshot_id"] for s in run.snapshots()) or "none"
            raise click.ClickException(
                f"no snapshot {snapshot_id!r} on {run_id}; it has: {have}"
            )
        interpretation = run.snapshot_interpretation(snap)
        click.echo(f"{snapshot_id}  ({snap.get('reason')})")
        click.echo(f"  taken:       {snap.get('taken_at')}")
        # The module's own terms: a granularity and its names for clustering, behaviors and their
        # cutoffs for a scan. Printing clustering's fields for both is what made a fully
        # calibrated scan read "granularity None / named 0".
        for column, cell in zip(interpretation.columns, interpretation.cells):
            click.echo(f"  {(column.lower() + ':'):<12} {cell}")
        click.echo(f"  code:        {snap.get('code_sha') or '—'}")
        click.echo(
            f"  used by:     {', '.join(used.get(snapshot_id, [])) or 'nothing yet'}"
        )
        artifact = run.snapshot_artifact(snapshot_id)
        click.echo(f"  instances:   {len(artifact.instances):,}")
        _echo_snapshot_clusters(run, snapshot_id)
        return

    snaps = run.snapshots()
    if not snaps:
        click.echo(f"{run_id} has pinned nothing yet.")
        return
    for snap in snaps:
        interpretation = run.snapshot_interpretation(snap)
        consumers = ", ".join(used.get(snap["snapshot_id"], [])) or "—"
        click.echo(
            f"{snap['snapshot_id']}  {str(snap.get('taken_at', ''))[:16]}  "
            f"{snap.get('reason'):<10}  {interpretation.headline or '—'}  "
            f"used by {consumers}"
        )


def _echo_snapshot_clusters(run, snapshot_id: str) -> None:
    """The labels as they were, read through the snapshot view (§21/§22).

    The CLI half of "a snapshot is readable": the GUI renders the frozen interpretation in the
    workspace, and this prints its labels, each with its path and the time it held, from the
    snapshot's own labels over the run's node cache. A module without labelled nodes (a scan's
    behaviors are already covered by its own columns above) prints nothing extra.
    """
    try:
        view = run.view(snapshot_id)
    except ValueError as exc:
        click.echo(f"  labels:      {exc}")
        return
    if view is None or not hasattr(view, "node_view"):
        return
    try:
        nodes = view.node_view()
        state = view.state()
    except FileNotFoundError as exc:
        # The cache is read from the run, not the snapshot; a deleted run takes it away, as
        # does a run fitted before the cache existed.
        click.echo(f"  labels:      unavailable ({exc})")
        return
    times = nodes.label_time(state.node_names)
    total = sum(times.values()) or 1
    paths = dict(state.resolved_paths)
    click.echo("  labels:")
    for node, time in sorted(times.items(), key=lambda item: -item[1]):
        shown = "unlabelled" if node == "unlabelled" else paths.get(node, node)
        click.echo(
            f"    {node:>10}  {time / 1e9:>9.1f} s  {100 * time / total:>5.1f}%  {shown}"
        )


@analysis.command("attach")
@click.argument("run_id")
@_EXPERIMENT_OPT
def analysis_attach(run_id: str, experiment_dir: Path) -> None:
    """Reattach to an analysis run — stream a live one's progress (it keeps running with
    the app/CLI closed) or print a finished one's status. Ctrl-C detaches; it does not
    cancel the run (use ``analysis cancel`` for that)."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.analysis_record(run_id)  # ClickException rather than a traceback
    except FileNotFoundError:
        raise click.ClickException(f"no analysis run {run_id!r} in {experiment_dir}")
    handle = proj.attach_analysis_run(run_id)
    _stream_progress(handle, "analysis")
    click.echo(f"status: {handle.status}")


@analysis.command("cancel")
@click.argument("run_id")
@_EXPERIMENT_OPT
def analysis_cancel(run_id: str, experiment_dir: Path) -> None:
    """Stop an active analysis run (kills its worker; no-op once terminal)."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.cancel_analysis(run_id)
    except FileNotFoundError:
        raise click.ClickException(f"no analysis run {run_id!r} in {experiment_dir}")
    click.echo(f"cancelled analysis: {run_id}")


@analysis.command("rebuild")
@click.argument("run_id")
@_EXPERIMENT_OPT
def analysis_rebuild(run_id: str, experiment_dir: Path) -> None:
    """Rebuild an open run's results from its stored output and its current settings.

    The repair for a run whose results are out of date — a change that did not finish writing
    them, so the settings moved and the results did not. `analysis status` says when a run is in
    that state. The expensive pass is untouched: this is one pass over the outputs, not another
    fit or search.
    """
    proj = _open_experiment(experiment_dir)
    run = proj.authored_run(run_id)
    if run is None:
        raise click.ClickException(
            f"analysis run {run_id!r} is not an authored run — there is nothing to rebuild "
            "from (a computed run is re-run, not re-derived)"
        )
    bar = _StageProgress()
    try:
        run.reproject(progress=bar)
    finally:
        bar.close()
    click.echo(f"rebuilt results for {run_id}")


@analysis.command("delete")
@click.argument("run_id")
@_EXPERIMENT_OPT
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def analysis_delete(run_id: str, experiment_dir: Path, yes: bool) -> None:
    """Delete an analysis run (its analysis/<run_id>/ directory)."""
    proj = _open_experiment(experiment_dir)
    if not yes:
        click.confirm(f"delete analysis run {run_id!r}?", abort=True)
    try:
        proj.delete_analysis(run_id)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"deleted analysis: {run_id}")


def _split_gap_ns(seconds: float | None) -> int | None:
    """A ``--split-gap`` in seconds as the ``t_ns`` the run records (Tenet 8)."""
    return None if seconds is None else _option_t_ns(seconds, "--split-gap")


def _clip_span_label(result: dict) -> str:
    """A clip's span for a report line — seconds once resolved, else the frames asked for
    (a frame span that resolves to no rows never becomes a time)."""
    if result.get("start_t_ns") is not None and result.get("end_t_ns") is not None:
        return f"{result['start_t_ns'] / 1e9:.2f}-{result['end_t_ns'] / 1e9:.2f}s"
    if result.get("start_frame") is not None:
        return f"frames {result['start_frame']}-{result.get('end_frame')}"
    return "(no span)"


def _needs_contiguous_exemplars(method_id: str | None) -> bool:
    """Whether ``method_id`` requires each example clip to sit inside one continuous run.

    An unknown method answers ``True``: screening runs before ``validate()``, which owns the
    "unknown query method" message and names the registered ones. Guessing permissively here
    would screen nothing and hand the user the wrong error.
    """
    from detail.pipeline.query import resolve_method

    if not method_id:
        return True
    try:
        method = resolve_method(method_id)
    except ValueError:
        return True
    return method.exemplars.requires_contiguous


def _screen_example_clips(
    experiment_dir, embed_run_id, groups, split_gap_ns, strict, *, method_id=None
):
    """Report every example clip that cannot be searched, and drop it unless ``strict``.

    A clip fails for four reasons — an unknown recording, a track this embed run never
    embedded, a span with no embedded frames, or a span crossing a break in tracking (rows
    exist only where the subject was detected, §9). Left to the runner, the first one raises
    and the rest are never mentioned; a CSV written elsewhere can easily carry several.
    Screening here names each, with its query, so the whole list can be fixed in one pass.

    The fourth reason is the searching method's to impose (``ExemplarBounds``), so
    ``method_id`` decides it; the modules that pass none match a path and keep the rule.
    """
    from detail.api.run_preview import ClipQueryPreview

    flat = [(name, spec) for name, specs in groups.items() for spec in specs]
    results = ClipQueryPreview(experiment_dir).check_examples(
        embed_run_id,
        [spec for _name, spec in flat],
        split_gap_ns,
        require_contiguous=_needs_contiguous_exemplars(method_id),
    )
    bad = [(name, spec, r) for (name, spec), r in zip(flat, results) if not r["ok"]]
    if not bad:
        return groups
    click.echo(
        f"{len(bad)} of {len(flat)} example clip(s) cannot be searched:", err=True
    )
    for name, spec, result in bad:
        click.echo(
            f"  [{name or 'query'}] {spec['recording_id']} track {spec['track_id']} "
            f"{_clip_span_label(result)} — {result['reason']}",
            err=True,
        )
    if strict:
        raise click.ClickException(
            f"{len(bad)} example clip(s) cannot be searched (--strict)"
        )
    click.echo("skipping them; pass --strict to fail instead.", err=True)
    kept = {name: [] for name in groups}
    for (name, spec), result in zip(flat, results):
        if result["ok"]:
            kept[name].append(spec)
    # A query whose every clip was dropped is dropped with them — leaving it empty would
    # fail validation ("no example clips") and bury the reasons just printed.
    emptied = [name for name, specs in kept.items() if groups[name] and not specs]
    for name in emptied:
        click.echo(
            f"dropping query {name or 'query'!r}: none of its clips are searchable.",
            err=True,
        )
        del kept[name]
    return kept


def _parse_method_params(pairs: tuple[str, ...]) -> dict:
    """``NAME=VALUE`` pairs into the ``method_params`` dict.

    Values stay **strings**; the method's own pydantic model types and bounds-checks them at
    ``validate()``, which is the single source (§6.5) and the only thing that knows what the
    chosen method takes. Typing them here would be a second copy of every field's type.

    The alternative — one generated ``@click.option`` per parameter — cannot work: ``--method``
    is chosen at parse time, so the option set would have to depend on a value click has not
    read yet. Each method's generated docs page lists what it takes.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            raise click.ClickException(
                f"bad --method-param {pair!r}; expected NAME=VALUE"
            )
        out[name.strip()] = value.strip()
    return out


def _parse_example(value: str, default_track: int) -> tuple[str | None, dict]:
    """Parse an ``--example`` spec into ``(query_name | None, example dict)``.

    The spec is ``'[query=]recording[:track]:start-end'`` — an optional ``query=`` prefix
    assigns the clip to that named query (unambiguous: no ``=`` in a spec). The span is
    **seconds** by default (``12.4-13.1`` → a t_ns span, Tenet 8) or **frames** with an
    ``f`` suffix (``1240f-1310f``); track defaults to ``default_track``."""
    query_name: str | None = None
    if "=" in value:
        prefix, value = value.split("=", 1)
        query_name = prefix.strip() or None
    parts = value.split(":")
    if len(parts) == 3:
        recording_id, track_text, span = parts
        try:
            track_id = int(track_text)
        except ValueError:
            raise click.ClickException(
                f"bad track {track_text!r} in --example {value!r}; a track is a whole "
                "number, e.g. 'recording:0:12.4-13.1'"
            ) from None
    elif len(parts) == 2:
        recording_id, span = parts
        track_id = default_track
    else:
        raise click.ClickException(
            f"bad --example {value!r}; expected '[query=]recording[:track]:start-end'"
        )
    try:
        start_text, end_text = span.split("-")
    except ValueError:
        raise click.ClickException(
            f"bad span in --example {value!r}; expected 'start-end'"
        )
    if start_text.endswith("f") and end_text.endswith("f"):
        try:
            return query_name, {
                "recording_id": recording_id,
                "track_id": track_id,
                "start_frame": int(start_text[:-1]),
                "end_frame": int(end_text[:-1]),
            }
        except ValueError:
            raise click.ClickException(f"bad frame span in --example {value!r}")
    try:
        return query_name, {
            "recording_id": recording_id,
            "track_id": track_id,
            "start_t_ns": seconds_to_t_ns(float(start_text), what="--example start"),
            "end_t_ns": seconds_to_t_ns(float(end_text), what="--example end"),
        }
    except ValueError:
        raise click.ClickException(
            f"bad span in --example {value!r}; use seconds (12.4-13.1) or frames "
            "(1240f-1310f)"
        )


def _query_methods() -> list[str]:
    """The §20 method ids this build offers, for `--method`'s choices.

    Read from the registry rather than a `Literal`, because §20's methods are a registry
    whose point is that one can be added without editing the module (`configs/query.py`),
    and filtered through §25 here — where the choices are *enumerated* — rather than at
    `resolve_method`, so a saved run naming a hidden method still opens and still reads.
    """
    from detail import surfaces
    from detail.core.registries import QUERY_METHODS
    from detail.query_methods import import_query_methods

    import_query_methods()
    return surfaces.surfaced(sorted(QUERY_METHODS))


@analysis.command("query")
@_EXPERIMENT_OPT
@click.option(
    "--search", "search_recordings", multiple=True, required=True,
    help="Recording id to search (repeatable; at least one).",
)  # fmt: skip
@click.option(
    "--embed-run", "embed_run_id", required=True,
    help=_QRY["embed_run_id"].description,
)  # fmt: skip
@click.option(
    "--behavior", "behavior_names", multiple=True,
    help="Declare a named behavior (repeatable, in order); attach clips with "
    "--example NAME=SPEC. Optional — a bare --example makes one default behavior.",
)  # fmt: skip
@click.option(
    "--example", "examples", multiple=True,
    help="Example clip '[behavior=]recording[:track]:start-end' — seconds (12.4-13.1) or "
    "frames with an f suffix (1240f-1310f). Repeatable.",
)  # fmt: skip
@click.option(
    "--from-csv", "from_csv",
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
    help="Also seed examples from a prior clip CSV, grouped by its query_name column — the "
    "same path as reusing a previous run's examples.",
)  # fmt: skip
@click.option(
    "--non-example", "non_examples", multiple=True,
    help="Non-example clip, same spec as --example. Only methods whose negatives are "
    "'optional' or 'required' can use them; the run is refused otherwise. Repeatable.",
)  # fmt: skip
@click.option(
    "--method", type=click.Choice(_query_methods()),
    default=_QRY["method"].default, show_default=True,
    help=_QRY["method"].description,
)  # fmt: skip
@click.option(
    "--method-param", "method_params", multiple=True, metavar="NAME=VALUE",
    help="A parameter of the chosen --method, e.g. --method-param k=30. "
    "Repeatable; the method's docs page lists what it takes (`detail docs`).",
)  # fmt: skip
@click.option(
    "--track", "track_id", type=_int_range(_QRY_DOC["track_id"]),
    default=_QRY["track_id"].default, show_default=True,
    help=_QRY["track_id"].description,
)  # fmt: skip
@click.option(
    "--split-gap", "trajectory_split_gap_s",
    type=_float_range(_QRY_DOC["trajectory_split_gap_s"]),
    default=_QRY["trajectory_split_gap_s"].default,
    help=_QRY["trajectory_split_gap_s"].description,
)  # fmt: skip
@click.option(
    "--min-match", "min_match_s",
    type=_float_range(_QRY_DOC["min_match_s"]),
    default=_QRY["min_match_s"].default, show_default=True,
    help=_QRY["min_match_s"].description,
)  # fmt: skip
@click.option(
    "--strict", is_flag=True, default=False,
    help="Fail if any example clip cannot be searched, instead of skipping it.",
)  # fmt: skip
def analysis_query(
    experiment_dir: Path,
    search_recordings: tuple[str, ...],
    embed_run_id: str,
    behavior_names: tuple[str, ...],
    examples: tuple[str, ...],
    non_examples: tuple[str, ...],
    from_csv: Path | None,
    method: str,
    method_params: tuple[str, ...],
    track_id: int,
    trajectory_split_gap_s: float | None,
    min_match_s: float,
    strict: bool,
) -> None:
    """Search a cohort for a behavior you can name, from a few example clips (§20).

    Writes one shard per (behavior x recording x subject) under analysis/<an_NN>/search/.
    The run stays **open** afterwards: there is no cutoff during the search, so choosing one
    costs nothing and never a second search. Calibration is the next step.
    """
    import collections

    proj = _open_experiment(experiment_dir)
    groups: "collections.OrderedDict[str | None, list[dict]]" = (
        collections.OrderedDict()
    )
    for name in behavior_names:
        groups.setdefault(name, [])
    for spec in examples:
        name, clip = _parse_example(spec, track_id)
        groups.setdefault(name, []).append(clip)
    if from_csv is not None:
        from detail.storage.clip_query_csv import read_clip_csv

        for row in read_clip_csv(from_csv):
            groups.setdefault(row.query_name, []).append(
                {
                    "recording_id": row.recording_id,
                    "track_id": row.track_id,
                    "start_t_ns": row.start_t_ns,
                    "end_t_ns": row.end_t_ns,
                }
            )
    if not any(groups.values()):
        raise click.ClickException("provide at least one --example or --from-csv")
    # Non-examples group by behavior the same way examples do, so `--non-example
    # rearing=rec:1.0-2.0` lands on the behavior it names. One naming no behavior with
    # examples would be a behavior with nothing to search *for*: the config cannot carry it,
    # so it is refused here rather than dropped.
    negatives: dict = {}
    for spec in non_examples:
        name, clip = _parse_example(spec, track_id)
        negatives.setdefault(name, []).append(clip)
    orphans = [name for name in negatives if not groups.get(name)]
    if orphans:
        shown = ", ".join("(no name)" if n is None else repr(n) for n in orphans)
        known = ", ".join(
            "(no name)" if n is None else repr(n)
            for n, clips in groups.items()
            if clips
        )
        raise UserInputError(
            f"--non-example names {shown}, which has no --example; non-examples attach "
            f"to a behavior that has examples: {known}"
        )
    split_gap_ns = _split_gap_ns(trajectory_split_gap_s)
    groups = _screen_example_clips(
        proj.dir, embed_run_id, groups, split_gap_ns, strict, method_id=method
    )
    if not any(groups.values()):
        raise click.ClickException("no usable example clips are left")

    config = {
        "source_recording_ids": list(search_recordings),
        "embed_run_id": embed_run_id,
        "track_id": track_id,
        "method": method,
        "method_params": _parse_method_params(method_params),
        "behaviors": [
            {
                "name": name or "behavior",
                "examples": clips,
                "negatives": negatives.get(name, []),
            }
            for name, clips in groups.items()
        ],
        "trajectory_split_gap_ns": split_gap_ns,
        "min_match_ns": _option_t_ns(min_match_s, "--min-match"),
    }
    try:
        run = proj.run_analysis("query", config)
    except (ValueError, KeyError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted query: {run.id}")
    _stream_progress(run, "query")
    run.wait()
    # Two orthogonal status planes (§21). `run.status` is the **executor's** — did the search
    # job finish — and is what a failure has to be read from. The **record's** status is
    # `open`, because an authored run stays open for its whole life and only the user closes
    # it. Reporting the executor's as though it were the run's would tell a user their run is
    # `complete` while the GUI correctly shows it open.
    if run.status not in ("complete", "partial"):
        raise click.ClickException(
            f"query {run.id!r} did not finish searching (status={run.status!r})"
            + _run_failure_reason(run)
        )
    record_status = proj.query_run(run.id).status
    click.echo(
        f"query run {run.id} searched (run is {record_status}) — set a cutoff per behavior "
        "to turn the results into instances:"
    )
    click.echo(f"  detail analysis query-calibrate --run {run.id} --curve q01")


@analysis.command("query-calibrate")
@_EXPERIMENT_OPT
@click.option("--run", "run_id", required=True, help="Query run id (e.g. an_01).")
@click.option(
    "--cutoff", "cutoffs", multiple=True, metavar="BEHAVIOR=CONFIDENCE",
    help="Set one behavior's confidence cutoff, e.g. q01=0.82. Repeatable. "
    "Use BEHAVIOR=none to clear one, which removes it from the artifact again.",
)  # fmt: skip
@click.option(
    "--name", "names", multiple=True, metavar="BEHAVIOR=NAME",
    help="Rename a behavior, e.g. q01='supported rearing'. Repeatable. The id never moves.",
)  # fmt: skip
@click.option(
    "--curve", "curve_for", default=None, metavar="BEHAVIOR",
    help="Print the instances-per-cutoff curve for one behavior and exit without editing.",
)  # fmt: skip
@click.option(
    "--steps", type=click.IntRange(2, 1001), default=21, show_default=True,
    help="Points on the --curve.",
)  # fmt: skip
@click.option(
    "--finalize", is_flag=True, default=False,
    help="Mark the run complete — your statement that you are done calibrating. Refused, "
    "naming the recordings, while the search has not reached every one.",
)  # fmt: skip
@click.option(
    "--reopen", is_flag=True, default=False,
    help="Move a finalized run back to open so it can be calibrated again.",
)  # fmt: skip
def analysis_query_calibrate(
    experiment_dir: Path,
    run_id: str,
    cutoffs: tuple[str, ...],
    names: tuple[str, ...],
    curve_for: str | None,
    steps: int,
    finalize: bool,
    reopen: bool,
) -> None:
    """Calibrate a §20 query run: set cutoffs, name behaviors, finalize.

    Every operation here is instant — it rewrites the run's authored state and reprojects the
    behavior artifact from search results already on disk. Nothing re-searches.

    A behavior with no cutoff is **absent** from the artifact, not present with zero
    instances: §19 reads a named behavior with no rows as "we looked and it never happened",
    which is a different claim from "nobody has chosen a threshold yet".
    """
    proj = _open_experiment(experiment_dir)
    run = proj.query_run(run_id)

    if curve_for:
        try:
            curve = run.frequency_curve(curve_for, steps=steps)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise click.ClickException(str(e))
        click.echo(f"{'cutoff':>8}  {'instances':>9}")
        for point in curve:
            flag = (
                "  (capped by the method's candidate limit)" if point["capped"] else ""
            )
            click.echo(f"{point['cutoff']:>8.3f}  {point['n_instances']:>9d}{flag}")
        click.echo(
            "\nInstance count is not guaranteed to fall as the cutoff rises: a method that "
            "re-forms extents can split one instance into two."
        )
        return

    if reopen:
        run.reopen()
        click.echo(f"reopened {run_id}")

    # Each edit reprojects — a shard read and a frame-view zarr per searched unit — which on a
    # cohort is long enough that a silent command is indistinguishable from a hang.
    bar = _StageProgress()
    try:
        for spec in names:
            behavior_id, _, name = spec.partition("=")
            if not name:
                raise click.ClickException(f"--name wants BEHAVIOR=NAME, got {spec!r}")
            run.name_behavior(behavior_id.strip(), name, progress=bar)
            click.echo(f"{behavior_id.strip()} is now {name!r}")
        for spec in cutoffs:
            behavior_id, _, raw = spec.partition("=")
            if not raw:
                raise click.ClickException(
                    f"--cutoff wants BEHAVIOR=CONFIDENCE, got {spec!r}"
                )
            value = None if raw.strip().lower() in ("none", "") else parse_finite(raw)
            run.set_cutoff(behavior_id.strip(), value, progress=bar)
            click.echo(
                f"{behavior_id.strip()} cutoff cleared — it is absent from the artifact again"
                if value is None
                else f"{behavior_id.strip()} cutoff {value}"
            )
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()

    if finalize:
        try:
            _with_stage_progress(lambda bar: run.finalize(progress=bar))
        except ValueError as e:  # a search that did not reach every recording
            raise click.ClickException(str(e))
        click.echo(f"finalized {run_id}")

    _echo_query_state(run)


def _echo_query_state(run) -> None:
    """One line per behavior, plus what the artifact currently states."""
    click.echo(f"\nrun {run.id} is {run.status}")
    for behavior in run.behaviors():
        cutoff = (
            "not calibrated" if behavior.cutoff is None else f"cutoff {behavior.cutoff}"
        )
        click.echo(f"  {behavior.id}  {behavior.name}  ({cutoff})")
    try:
        output = run.artifact()
    except FileNotFoundError:
        click.echo("  (no behavior output yet — the search did not finish)")
        return
    if run.artifact_stale:
        # The cutoffs above are the saved ones; the counts below are from before the last
        # change, whose rebuild failed. Printed side by side unmarked, they read as one state.
        click.echo(
            "  results: out of date — the last change was saved but not rebuilt; the counts "
            f"below are from before it. Rebuild with 'detail analysis rebuild {run.id}'"
        )
    click.echo(
        f"  behavior output: {output.bout_count} instance(s) over "
        f"{len(output.coverage)} searched unit(s), vocabulary "
        f"{output.manifest['vocabulary'] or '(nothing calibrated yet)'}"
    )


@analysis.command("audit")
@_EXPERIMENT_OPT
@click.option("--run", "run_id", required=True, help="Authored run id (e.g. an_01).")
@click.option(
    "--behavior", "label", default=None, metavar="BEHAVIOR",
    help=(
        "Report one behavior: a query behavior's id, or a clustering label's path "
        "(rearing/supported). Omit for the whole run — on a clustering run, every label."
    ),
)  # fmt: skip
@click.option(
    "--cutoff", type=FINITE_FLOAT, default=None,
    help="Report at this confidence cutoff instead of the run's own.",
)  # fmt: skip
@click.option(
    "--by-recording", is_flag=True, default=False,
    help="Split the estimate per recording — one cutoff can behave differently per animal.",
)  # fmt: skip
@click.option(
    "--by-cluster", is_flag=True, default=False,
    help="Clustering runs: under each label, each cluster's precision and 95% confidence "
    "interval.",
)  # fmt: skip
@click.option(
    "--goal-width", type=FINITE_FLOAT, default=None,
    help="Say whether each 95% confidence interval is this narrow yet (the GUI's default is "
    "0.20, i.e. ±0.10), and roughly where another round of 25 would put it — a sample-size "
    "goal, not a judgment of the label. Nothing is said about width without it.",
)  # fmt: skip
@click.option(
    "--own-only", is_flag=True, default=False,
    help="Clustering runs: a label's own clusters only, without its sub-labels' — the row "
    "INTERPRET's labels panel shows as the label's \"other\".",
)  # fmt: skip
def analysis_audit(
    experiment_dir: Path,
    run_id: str,
    label: str | None,
    cutoff: float | None,
    by_recording: bool,
    by_cluster: bool,
    goal_width: float | None,
    own_only: bool,
) -> None:
    """Read the precision a run's reliability audit supports (§19).

    Reads only. The judging itself is a blind review of video clips, which is a GUI loop
    (`docs/cli.md`, "Not in the CLI") — but the *number* that review buys belongs to anyone
    scripting against the run, which is what this is for.

    Precision comes with its interval and its `n`, because 1.00 from n=3 and 1.00 from n=50
    are the same figure and different claims. It is withheld entirely, with a reason, when the
    audit no longer covers enough of the run to speak for it.

    A clustering run's labels nest, so it is addressed by label path, and with no `--behavior`
    it prints every label — a parent over everything under it, then each sub-label.

    A temporal clustering run's precision comes from its clip review (`clip_review.csv`: every
    clip of a cluster reviewed, written by `finish_review`), printed first with each cluster's
    figure; the earlier one-by-one audit (`audit.csv`) follows under its own heading, read as it
    always was.
    """
    proj = _open_experiment(experiment_dir)
    run = proj.authored_run(run_id)
    if run is None:
        raise click.ClickException(
            f"{run_id} is not an authored run (a query or clustering run), so it has no "
            "reliability audit. `detail analysis list` shows each run's module."
        )
    if hasattr(run, "review_precision"):
        _echo_clip_review(run, label)
        click.echo("Earlier one-by-one audit:")
    if run.nested_labels:
        paths = [label] if label else run.audit_labels()
        scopes = [(path, {"path": path, "own_only": own_only}) for path in paths]
    else:
        scopes = [(label or "all behaviors", {"label": label})]
    if not scopes:
        click.echo("(nothing labelled — no label to audit yet)")
        return
    from detail.storage import audit_figures
    from detail.storage import behavior_audit
    from detail.storage.behavior_output import PATH_SEPARATOR as sep

    goal = behavior_audit.DEFAULT_GOAL_WIDTH if goal_width is None else goal_width
    for scope, address in scopes:
        estimate = run.audit_precision(cutoff=cutoff, **address)
        if estimate is None:
            click.echo("(not audited — no reliability estimate for this run)")
            return
        depth = scope.count(sep) if run.nested_labels else 0
        shown = "  " * depth + scope.rsplit(sep, 1)[-1]
        if estimate.get("reason"):
            click.echo(f"{shown}: no estimate — {estimate['reason']}")
            continue
        notes = audit_figures.notes(estimate)
        if goal_width is not None and estimate.get("interval"):
            notes.append(_width_note(estimate, goal))
        suffix = f"  [{'; '.join(notes)}]" if notes else ""
        click.echo(f"{shown}: {audit_figures.line(estimate)}{suffix}")
        if by_cluster and run.nested_labels:
            for row in run.audit_by_cluster(path=scope, own_only=own_only):
                if row["path"] != scope and row["path"] in paths:
                    continue  # a sub-label printed below lists its own clusters
                width = (
                    f"  [{_width_note(row['estimate'], goal)}]"
                    if goal_width is not None and row["estimate"].get("interval")
                    else ""
                )
                click.echo(
                    f"  {'  ' * depth}#{row['node']:<6} {row['size']} instances, "
                    f"{row['share']:.0%} of the label  {audit_figures.line(row['estimate'])}"
                    f"{width}"
                )
        if by_recording:
            rows = run.audit_precision_by_recording(cutoff=cutoff, **address) or []
            for row in rows:
                click.echo(
                    f"  {'  ' * depth}{row['recording_id']:<24} {audit_figures.line(row)}"
                )


def _review_paths(run) -> list[str]:
    """Every label path a clip review can address: each label, and each label above it."""
    from detail.storage.behavior_output import PATH_SEPARATOR as sep

    paths: set[str] = set()
    for path in run.state().resolved_paths.values():
        parts = path.split(sep)
        for depth in range(1, len(parts) + 1):
            paths.add(sep.join(parts[:depth]))
    return sorted(paths, key=lambda p: p.split(sep))


def _echo_clip_review(run, label: str | None) -> None:
    """A temporal clustering run's clip review, label by label and cluster by cluster (§19/§22).

    A cluster is reviewed whole (every clip watched, `finish_review`), so a reviewed cluster's
    figure is exact. A label's figure combines its clusters' by size and is printed only once
    every cluster under it is reviewed, as AUDIT shows it; until then the line says how many are.
    """
    from detail.storage import audit_figures
    from detail.storage import behavior_audit as audit
    from detail.storage.behavior_output import PATH_SEPARATOR as sep

    click.echo("Clip review (every clip of a cluster reviewed):")
    paths = [label] if label else _review_paths(run)
    if not paths:
        click.echo("  (nothing labelled — no label to review yet)")
        return
    try:
        clusters = run.review_by_cluster()
        estimates = [(path, run.review_precision(path)) for path in paths]
    except FileNotFoundError as exc:
        click.echo(f"  unavailable ({exc})")
        return
    for path, estimate in estimates:
        depth = path.count(sep)
        shown = "  " * (depth + 1) + path.rsplit(sep, 1)[-1]
        under = [row for row in clusters if audit.under(row["path"], path)]
        done = [row for row in under if audit_figures.complete(row["estimate"])]
        if estimate is not None and audit_figures.complete(estimate):
            click.echo(
                f"{shown}: {audit_figures.line(estimate)}, every cluster reviewed "
                f"({len(under)})"
            )
        else:
            click.echo(f"{shown}: {len(done)} of {len(under)} clusters reviewed")
        for row in under:
            if row["path"] != path:
                continue  # a sub-label lists its own clusters under its own line
            figure = (
                audit_figures.line(row["estimate"])
                if audit_figures.complete(row["estimate"])
                else "not reviewed"
            )
            click.echo(
                f"  {'  ' * (depth + 1)}#{row['node']:<6} {row['size']} clips, "
                f"{row['share']:.0%} of the label  {figure}"
            )


def _width_note(estimate: dict, goal: float) -> str:
    """How wide the interval is against the goal width, and roughly where another round would
    put it — sample-size facts, not a verdict."""
    from detail.storage import audit_figures
    from detail.storage import behavior_audit

    width = audit_figures.width(estimate)
    if audit_figures.goal_reached(estimate, goal):
        return f"CI width {width:.2f}, goal {goal:.2f} reached"
    after = behavior_audit.projected_interval(estimate, behavior_audit.DEFAULT_BUDGET)
    projection = (
        f"; {behavior_audit.DEFAULT_BUDGET} more would give about "
        f"{after[0]:.2f}–{after[1]:.2f}"
        if after
        else ""
    )
    return f"CI width {width:.2f}, goal {goal:.2f}{projection}"


@analysis.command("export-query")
@_EXPERIMENT_OPT
@click.option("--run", "run_id", required=True, help="Query run id (e.g. an_01).")
@click.option(
    "--out", "out_dir", required=True, type=UserPath(file_okay=False, path_type=Path),
    help=(
        "Directory to create the export folder in; each export makes a new one holding a "
        "behavior-stream CSV per searched recording."
    ),
)  # fmt: skip
def analysis_export_query(experiment_dir: Path, run_id: str, out_dir: Path) -> None:
    """Export a query run's calibrated behaviors as per-recording behavior-stream CSVs.

    Each row carries ``origin: predicted``, so re-importing one as a behavior stream can never
    be read back as ground truth. A run whose search did not reach every recording is refused,
    naming the ones it missed.
    """
    proj = _open_experiment(experiment_dir)
    try:
        run = proj.query_run(run_id)
        written = _with_stage_progress(lambda bar: run.export(out_dir, progress=bar))
        pinned = run.snapshots()
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    click.echo(f"exported to {written}")
    if pinned:
        # The id the export corresponds to, so a figure built from these CSVs can cite something
        # that is still on disk after the run is reopened and re-cut (§21).
        click.echo(f"  pinned as {pinned[-1]['snapshot_id']}")


def _temporal_method_fields(method_id: str) -> set[str]:
    """The parameter names one temporal method actually accepts.

    Read off the method's own ``config_model`` rather than listed here, so a third method's knobs
    are recognised the day it registers and nothing has to be kept in step by hand.
    """
    from detail.pipeline.temporal_clustering import resolve_method

    return set(resolve_method(method_id).config_model.model_fields)


def _temporal_cluster_methods() -> list[str]:
    """The registered Section 22 temporal-cluster method ids, for ``--method``'s choices.

    Read rather than hardcoded. A *separate* registry from the pose one, which is empty since pose
    clustering was removed: Section 22 keeps the two apart.
    """
    from detail.cluster_methods import import_temporal_cluster_methods
    from detail.core.registries import TEMPORAL_CLUSTER_METHODS

    import_temporal_cluster_methods()
    return sorted(TEMPORAL_CLUSTER_METHODS)


#: Nodes ``--tree`` prints before it stops. A real fit has thousands, coarsest first, and the
#: coarse end is where the decisions are — a full dump is a screen of leaves nobody reads.
_TREE_ROWS = 60


def _pose_names(pairs: tuple[str, ...]) -> dict[str, str]:
    """``("3=rearing", "7=rearing")`` -> ``{"3": "rearing", "7": "rearing"}``.

    Several clusters sharing one name is the normal case (Section 22), so nothing here objects
    to a repeated value. ``CLUSTER=`` with an empty name clears one.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        cluster, sep, name = pair.partition("=")
        if not sep:
            raise click.ClickException(
                f"--name-node expects NODE=NAME, got {pair!r}; e.g. --name-node 118='rearing'"
            )
        out[cluster.strip()] = name.strip()
    return out


def _echo_coverage(proj, run, inside: str) -> None:
    """How much of each video was analysed, and how much of that the run labels.

    Printed after whatever the command did, so an edit reports the coverage it produced. A gap is
    invisible in a timeline at any realistic length; these two numbers are what makes it knowable.
    """
    from detail.api import authored_run

    try:
        rows = run.coverage_summary()
    except FileNotFoundError:
        return
    for line in authored_run.coverage_lines(rows, proj.recording_names(), inside):
        click.echo(f"coverage: {line}")


def _interpret_options(kind: str):
    """The options ``temporal-interpret`` takes, in one place (``pose-interpret`` shared them
    until pose clustering was removed)."""

    def decorate(fn):
        options = [
            _EXPERIMENT_OPT,
            click.option(
                "--run",
                "run_id",
                required=True,
                help=f"{kind.capitalize()}-clustering run id (e.g. an_01).",
            ),
            click.option(
                "--tree",
                "show_tree",
                is_flag=True,
                default=False,
                help="Print the merge tree, coarsest first: every node with its member time, "
                "its share of the run, how many clips it has and its label, then exit.",
            ),
            click.option(
                "--name-node",
                "node_names",
                multiple=True,
                metavar="NODE=NAME",
                help="Label a node of the merge tree, e.g. 118='immobility'. The label covers "
                "everything beneath it unless a node inside has its own, more specific one. A "
                "name that already exists adds the node to that label. Repeatable; use NODE= to "
                "remove a node's label.",
            ),
            click.option(
                "--from-screen",
                "from_screen",
                default=None,
                metavar="CMP_ID",
                help="Record that the nodes labelled with --name-node were found by this "
                "saved difference screen (a comparison with --source nodes), so a later "
                "comparison of the label over the same column can say the screen chose the "
                "cluster it scores. Each node must be one the screen kept, or sit inside or "
                "above one.",
            ),
            click.option(
                "--at-p",
                "at_p",
                type=FINITE_FLOAT,
                default=None,
                metavar="LEVEL",
                help="With --from-screen: the strictness the screen was read at, a family-wise "
                "level in (0, 1) — the GUI's Strictness slider. Its kept set is then the "
                "clusters clearing the floor at p <= LEVEL, and the label records the level. "
                "Default: the screen's own 5% floor.",
            ),
            click.option(
                "--rename-label",
                "renames",
                multiple=True,
                metavar="OLD=NEW",
                help="Rename a label, and with it every node that carries it. Repeatable.",
            ),
            click.option(
                "--merge-label",
                "merges",
                multiple=True,
                metavar="SOURCE=INTO",
                help="Move every node of label SOURCE to label INTO and remove SOURCE. "
                "Repeatable.",
            ),
            click.option(
                "--rebuild-cache",
                is_flag=True,
                default=False,
                help="Rebuild the node cache (every node's clips and figures) from the "
                "run's fit: for a cache reported stale, or one built before every clip was "
                "listed.",
            ),
            click.option(
                "--finalize",
                is_flag=True,
                default=False,
                help="Mark the run complete, your statement that you are done interpreting.",
            ),
            click.option(
                "--reopen",
                is_flag=True,
                default=False,
                help="Move a finalized run back to open so it can be interpreted again.",
            ),
        ]
        for option in reversed(options):
            fn = option(fn)
        return fn

    return decorate


def _interpret(
    proj,
    run,
    *,
    show_tree,
    node_names,
    rebuild_cache,
    finalize,
    reopen,
    inside,
    renames=(),
    merges=(),
    from_screen=None,
    at_p=None,
) -> None:
    """The body both interpret commands run, against either kind of run handle (§22)."""
    failed = False
    try:
        if from_screen and not node_names:
            raise click.UsageError(
                "--from-screen records where --name-node labels came from"
            )
        if at_p is not None and not from_screen:
            raise click.UsageError("--at-p is the strictness --from-screen is read at")
        if at_p is not None and not 0.0 < at_p < 1.0:
            raise click.UsageError(
                f"--at-p is a family-wise level in (0, 1), got {at_p!r}"
            )
        note = getattr(run, "subsample_note", lambda: None)()
        if note:
            # Printed before anything else and on every invocation: a capped fit is identical to
            # a full one in every other respect, so the one place it can be noticed is here.
            click.echo(f"note: {note}")
        if rebuild_cache:
            _with_stage_progress(lambda bar: run.rebuild_node_cache(progress=bar))
            click.echo("node cache rebuilt")
        if show_tree:
            _echo_tree(run)
            return
        # Every pair is read before anything is written: a malformed one found after `--name`
        # had already landed left the run half edited. And the names, renames and merges are one
        # edit (`edit_labels`), so a label name that is wrong — found only against the labels
        # the edits before it leave — writes nothing either.
        rename_pairs = _label_pairs(renames, "--rename-label", "OLD=NEW")
        merge_pairs = _label_pairs(merges, "--merge-label", "SOURCE=INTO")
        names = _pose_names(node_names) if node_names else None
        origins = _label_origins(proj, run, names, from_screen, at_p) if names else None
        if names or rename_pairs or merge_pairs:
            extra = {"origins": origins} if origins else {}
            # The edit re-projects the whole run's output, so it shows its progress.
            _with_stage_progress(
                lambda bar: run.edit_labels(
                    names,
                    renames=rename_pairs,
                    merges=merge_pairs,
                    progress=bar,
                    **extra,
                )
            )
        if names:
            at = f" at p <= {at_p:g}" if at_p is not None else ""
            click.echo(
                f"labelled {len(node_names)} node(s)"
                + (f", from screen {from_screen}{at}" if from_screen else "")
            )
        for old, new in rename_pairs:
            click.echo(f"renamed {old!r} to {new!r}")
        for source, into in merge_pairs:
            click.echo(f"merged {source!r} into {into!r}")
        if finalize:
            _with_stage_progress(lambda bar: run.finalize(progress=bar))
            click.echo("run finalized")
        if reopen:
            run.reopen()
            click.echo("run reopened")
    except (KeyError, ValueError, FileNotFoundError) as e:
        failed = True
        raise click.ClickException(str(e))
    finally:
        if not failed:
            _echo_coverage(proj, run, inside)


def _label_origins(
    proj, run, names: dict, from_screen: str | None, at_p: float | None = None
) -> dict | None:
    """``{node: LabelOrigin}`` for the nodes ``--name-node`` labels, from the screen
    ``--from-screen`` names read at the strictness ``--at-p`` (its own floor for ``None``), or
    ``None`` without it. A node the screen did not point at is refused naming the clusters it
    kept: an origin that is not true is worse than none.
    """
    if not from_screen:
        return None
    from detail.api import comparison as comparison_api

    named = [node for node, name in names.items() if name]
    origins = comparison_api.label_origins(
        proj,
        from_screen,
        [int(n) for n in named],
        ancestors_of=run.ancestors_of,
        alpha=at_p,
    )
    out = {}
    for node in named:
        origin = origins[int(node)]
        if origin is None:
            kept = proj.get_comparison(from_screen).kept_nodes(at_p)
            at = f" at p <= {at_p:g}" if at_p is not None else ""
            raise click.ClickException(
                f"node {node} is not one screen {from_screen} kept{at}, nor inside or above "
                f"one; it kept {', '.join(str(k) for k in kept) or 'nothing'}"
            )
        out[node] = origin
    return out


def _label_pairs(pairs, option: str, shape: str) -> list[tuple[str, str]]:
    """``A=B`` pairs from a repeatable option, each side stripped and required."""
    out = []
    for pair in pairs:
        left, sep, right = pair.partition("=")
        if not sep or not left.strip() or not right.strip():
            raise click.UsageError(f"{option} takes {shape}, got {pair!r}")
        out.append((left.strip(), right.strip()))
    return out


def _echo_tree(run) -> None:
    """Every node, coarsest level first, with what its card would say.

    Breadth-first, so a listing cut at ``_TREE_ROWS`` shows the top of the tree whole rather
    than one deep branch: the coarse nodes are where a first label usually goes.
    """
    try:
        view = run.node_view()
    except FileNotFoundError:
        raise click.ClickException(
            "this run has no node cache yet; build one with --rebuild-cache"
        )
    names = run.node_names()  # legacy names folded on read
    origins = run.state().label_origins
    rows: list[tuple[int, int]] = []
    level = [view.root]
    depth = 0
    while level:
        rows.extend((node, depth) for node in level)
        level = [
            child
            for node in level
            for child in sorted(
                view.children(node), key=lambda c: -view.summary(c).time_ns
            )
        ]
        depth += 1
    click.echo("   node  parent  depth       time   share  clips  label")
    for node, depth in rows[:_TREE_ROWS]:
        summary = view.summary(node)
        own = names.get(str(node), "")
        inherited = next(
            (
                names[str(a)]
                for a in reversed(view.ancestors(node))
                if names.get(str(a))
            ),
            "",
        )
        shown = own or (f"({inherited})" if inherited else "-")
        if own and str(node) in origins:
            shown += f" [from {origins[str(node)].get('comparison', '?')}]"
        parent = "-" if summary.parent < 0 else str(summary.parent)
        click.echo(
            f"  {node:>5}  {parent:>6}  {depth:>5}  {summary.time_ns / 1e9:>8.1f}s  "
            f"{100 * summary.share:5.1f}%  {len(view.clips(node)):>5}  {shown}"
        )
    if len(rows) > _TREE_ROWS:
        click.echo(f"  … {len(rows) - _TREE_ROWS:,} more, finer than these")
    click.echo(
        "  (label) is covered by a parent's label; [from cmp_NN] a label a difference screen "
        "chose; clips: actions, one unbroken stretch at the finest level each, at least as "
        "long as the run's shortest instance"
    )


@analysis.command("temporal-clustering")
@_EXPERIMENT_OPT
@click.option(
    "--recording", "recordings", multiple=True, required=True,
    help="Recording id to fit the vocabulary on (repeatable; at least one).",
)  # fmt: skip
@click.option(
    "--embed-run", "embed_run_id", default=None,
    help=_TEMPORAL["embed_run_id"].description,
)  # fmt: skip
@_EMBED_REF_OPT
@click.option("--track", "track_id", type=int, default=0, show_default=True,
              help=_TEMPORAL["track_id"].description)  # fmt: skip
@click.option(
    "--method", default=_TEMPORAL["method"].default,
    type=click.Choice(_temporal_cluster_methods()),
    show_default=True, help=_TEMPORAL["method"].description,
)  # fmt: skip
@click.option(
    "--detector", default=None, type=click.Choice(["binseg", "novelty", "speed", "abd"]),
    help="Transition detector that cuts each trajectory into bouts. Left unset, the method "
    "default (binseg, which measured better than novelty) is used.",
)  # fmt: skip
@click.option(
    "--min-bout", "min_len_s", type=FINITE_FLOAT, default=None,
    help="Drop bouts shorter than this many seconds. Left unset, the method default is used.",
)  # fmt: skip
@click.option(
    "--band", "band_s", type=FINITE_FLOAT, default=None,
    help="DTW warp tolerance in seconds (0 = rigid). Left unset, the method default is used.",
)  # fmt: skip
@click.option(
    "--max-bouts", "max_segments", type=int, default=None,
    help="dtw_finch: cap the bouts compared against each other, evenly subsampled. The "
    "comparison is O(bouts squared) and this is the only bound on it. Every other bout is still "
    "labeled afterwards. Omit to compare every bout.",
)  # fmt: skip
@click.option("--fit-rows", type=int, default=None, hidden=True)
@click.option(
    "--fit-share", type=FINITE_FLOAT, default=None,
    help="codebook_viterbi: " + _CODEBOOK["fit_share"].description
    + " Left unset, the method default is used.",
)  # fmt: skip
@click.option(
    "--codewords", type=int, default=None,
    help="codebook_viterbi: " + _CODEBOOK["codewords"].description
    + " Left unset, the method default is used.",
)  # fmt: skip
@click.option(
    "--min-run", "min_run_s", type=FINITE_FLOAT, default=None,
    help="codebook_viterbi: runs shorter than this many seconds are dropped (not merged "
    "into a neighbour) rather than becoming instances. Left unset, the method default is "
    "used.",
)  # fmt: skip
def analysis_temporal_clustering(
    experiment_dir: Path,
    recordings: tuple[str, ...],
    embed_run_id: str | None,
    embed_refs: tuple[str, ...],
    track_id: int,
    method: str,
    detector: str | None,
    min_len_s: float | None,
    band_s: float | None,
    max_segments: int | None,
    fit_rows: int | None,
    fit_share: float | None,
    codewords: int | None,
    min_run_s: float | None,
) -> None:
    """Discover a vocabulary of behaviors across a recording selection (Section 22).

    Each trajectory is cut into bouts, every pair of bouts is compared, and the comparison is
    grouped into a vocabulary. A cluster is a kind of movement and an instance is a bout.

    The fit is one job and it leaves the run **open**, not complete: a vocabulary is not a
    result until somebody has looked at it and named things. Interpret it with
    ``analysis temporal-interpret``.

    The comparison is quadratic in the number of bouts. ``--max-bouts`` bounds it, and a run
    fitted that way records what it left out so no result is quietly built on a fraction.
    """
    if fit_rows is not None:
        # Refused rather than converted: a row total has no equivalent share without knowing the
        # recordings' lengths, and it also set the codeword count, which is now its own setting.
        raise click.UsageError(
            "--fit-rows was removed: the codebook sample is now a share of each recording "
            "(--fit-share, percent) and the codeword count is its own setting (--codewords)."
        )
    config: dict = {
        "source_recording_ids": list(recordings),
        "track_id": track_id,
        "method": method,
    }
    # Only the knobs actually given are passed, so an unset option means "the method default"
    # rather than a None the model would reject.
    given = {
        "detector": detector,
        "min_len_s": min_len_s,
        "band_s": band_s,
        "max_segments": max_segments,
        "fit_share": fit_share,
        "codewords": codewords,
        "min_run_s": min_run_s,
    }
    method_params = {k: v for k, v in given.items() if v is not None}
    # A knob the chosen method does not have is **refused**, not dropped. Pydantic ignores an
    # unknown field by default, so `--band 0.5` under a method with no DTW would silently do
    # nothing and the run would look like it had honoured the flag.
    known = set(_temporal_method_fields(method))
    unknown = sorted(set(method_params) - known)
    if unknown:
        raise click.ClickException(
            f"{', '.join(unknown)} is not a parameter of method {method!r}; it accepts "
            f"{', '.join(sorted(known))}"
        )
    if method_params:
        config["method_params"] = method_params

    proj = _open_experiment(experiment_dir)
    config.update(_embedding_selection(proj, embed_run_id, embed_refs))
    try:
        run = proj.run_analysis("temporal_clustering", config)
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted temporal clustering: {run.id}")
    _stream_progress(run, "temporal clustering")
    _check_status(run, "temporal clustering")
    click.echo(f"\n  detail analysis temporal-interpret --run {run.id} --tree")


@analysis.command("temporal-interpret")
@_interpret_options("temporal")
def analysis_temporal_interpret(
    experiment_dir: Path,
    run_id: str,
    show_tree: bool,
    node_names: tuple[str, ...],
    renames: tuple[str, ...],
    merges: tuple[str, ...],
    rebuild_cache: bool,
    finalize: bool,
    reopen: bool,
    from_screen: str | None,
    at_p: float | None,
) -> None:
    """Interpret a Section 22 temporal-clustering run: label nodes of its tree, finish.

    A label on a node of the tree of behaviors covers everything beneath it unless a node inside
    has its own. Labelling rewrites the run's authored state and its output from the comparison
    on disk; nothing re-compares. ``--tree`` shows the nodes.
    """
    proj = _open_experiment(experiment_dir)
    _interpret(
        proj,
        proj.temporal_clustering_run(run_id),
        show_tree=show_tree,
        node_names=node_names,
        renames=renames,
        merges=merges,
        rebuild_cache=rebuild_cache,
        finalize=finalize,
        reopen=reopen,
        inside="in a bout",
        from_screen=from_screen,
        at_p=at_p,
    )


@analysis.command("export-temporal-clustering")
@_EXPERIMENT_OPT
@click.option(
    "--run", "run_id", required=True, help="Temporal-clustering run id (e.g. an_01)."
)
@click.option(
    "--out", "out_dir", required=True, type=UserPath(file_okay=False, path_type=Path),
    help=(
        "Directory to create the export folder in; each export makes a new one holding a "
        "behavior-stream CSV per fitted recording."
    ),
)  # fmt: skip
def analysis_export_temporal_clustering(
    experiment_dir: Path, run_id: str, out_dir: Path
) -> None:
    """Export a temporal-clustering run's vocabulary as per-recording behavior-stream CSVs.

    Each row carries ``origin: predicted``, so re-importing one as a behavior stream can never
    be read back as ground truth.
    """
    proj = _open_experiment(experiment_dir)
    try:
        run = proj.temporal_clustering_run(run_id)
        written = _with_stage_progress(lambda bar: run.export(out_dir, progress=bar))
        pinned = run.snapshots()
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    click.echo(f"exported to {written}")
    if pinned:
        # The id the export corresponds to, so a figure built from these CSVs can cite something
        # that is still on disk after the run is reopened and re-cut (§21).
        click.echo(f"  pinned as {pinned[-1]['snapshot_id']}")


@analysis.command("cluster-montage")
@_EXPERIMENT_OPT
@click.option(
    "--run",
    "run_id",
    required=True,
    help="Temporal-clustering run id (e.g. an_01).",
)
@click.option(
    "--node",
    type=int,
    required=True,
    help="The node of the run's tree (as `temporal-interpret --tree` lists them).",
)
@click.option(
    "--out",
    "out_path",
    type=UserPath(dir_okay=False, path_type=Path),
    default=None,
    help="Write the montage itself: an .mp4 or .gif.",
)
@click.option(
    "--per-clip-gifs",
    "gif_dir",
    type=UserPath(file_okay=False, path_type=Path),
    default=None,
    help="Write one GIF per clip into this folder, named rank_recording_m-ss.t.gif.",
)
@click.option(
    "--csv",
    "csv_path",
    type=UserPath(dir_okay=False, path_type=Path),
    default=None,
    help="Write the clips as a CSV list: a clip-query CSV (`detail analysis query "
    "--from-csv` reads it as example clips) with rank, recording name, subject, times in "
    "seconds, node and run after the standard columns.",
)
@click.option(
    "--raw/--no-raw",
    default=_MONTAGE_DEFAULTS.raw,
    show_default=True,
    help="Draw the footage, with the animal outlined. Without it, masks and keypoints are "
    "drawn on white.",
)
@click.option(
    "--mask/--no-mask",
    default=_MONTAGE_DEFAULTS.mask,
    show_default=True,
    help="Draw the clip's animal: outlined on the footage, filled on white.",
)
@click.option(
    "--keypoints/--no-keypoints",
    default=_MONTAGE_DEFAULTS.keypoints,
    show_default=True,
    help="Draw the keypoints the embedding run used.",
)
@click.option(
    "--mask-trail/--no-mask-trail",
    default=None,
    help="Draw the outline's last half second, fading. Default: on with --mask.",
)
@click.option(
    "--keypoint-trail/--no-keypoint-trail",
    default=_MONTAGE_DEFAULTS.keypoint_trail,
    show_default=True,
    help="Draw each keypoint's path over the last half second (needs --keypoints).",
)
@click.option(
    "--trail-point",
    "trail_points",
    multiple=True,
    help="A keypoint that leaves a trail, by name. Repeatable; default every point.",
)
@click.option(
    "--aligned/--as-recorded",
    default=_MONTAGE_DEFAULTS.aligned,
    show_default=True,
    help="Turn each clip so the animal's body points up, frame by frame, from the keypoint "
    "run the embedding read and its heading definition. A frame with no body axis is drawn "
    "as recorded and marked.",
)
@click.option(
    "--context",
    "context_s",
    type=finite_float_range(0.0, 60.0),
    default=0.0,
    show_default=True,
    help="Seconds of footage either side of each clip: each tile plays the clip and its "
    "context. Labels still name the clip's own start.",
)
@click.option(
    "--clips",
    "limit",
    type=click.IntRange(min=1),
    default=_MONTAGE_CLIPS,
    show_default=True,
    help="How many clips, counted from the top of --sort's order.",
)
@click.option(
    "--sort",
    type=click.Choice(list(_MONTAGE_SORTS)),
    default=_MONTAGE_SORT,
    show_default=True,
    help="The clips' order: most typical first, least typical first, by recording, "
    "longest first, or random (the same shuffle for a node every time).",
)
@click.option(
    "--blind",
    is_flag=True,
    default=False,
    help="Label recordings A, B, C instead of by name (GIF names too).",
)
@click.option(
    "--max-width",
    type=click.IntRange(64, 7680),
    default=None,
    help="Width in pixels of --out or each --per-clip-gifs GIF. Default: 640 for a GIF, "
    "200-pixel tiles for an mp4.",
)
@click.option(
    "--fps",
    type=finite_float_range(1.0, 60.0),
    default=None,
    help="Frames a second of --out or each --per-clip-gifs GIF. Default: 12 for a GIF, the "
    "montage's own clock for an mp4.",
)
def analysis_cluster_montage(
    experiment_dir: Path,
    run_id: str,
    node: int,
    out_path: Path | None,
    gif_dir: Path | None,
    csv_path: Path | None,
    raw: bool,
    mask: bool,
    keypoints: bool,
    mask_trail: bool | None,
    keypoint_trail: bool,
    trail_points: tuple[str, ...],
    aligned: bool,
    context_s: float,
    limit: int,
    sort: str,
    blind: bool,
    max_width: int | None,
    fps: float | None,
) -> None:
    """Write a cluster montage: a node's clips side by side, one GIF per clip, or a CSV list.

    An export of a node's clips, the same files a clip grid's Export ▾ writes in the app
    (Section 22): 25 clips unless --clips says otherwise, most typical first unless --sort says
    otherwise, each cropped to its animal and labelled `#rank · recording · start`, all starting
    together and each looping from its start the moment it ends. Drawn by default as the mask
    and its trail, on white. Give at least one of --out, --per-clip-gifs and --csv.
    """
    from detail.api.animation import GIF_DEFAULT_FPS, GIF_DEFAULT_WIDTH
    from detail.api.clip_montage import (
        ClusterMontage,
        MontageLayers,
        clip_file_stem,
        cluster_montage,
        existing_gifs,
        montage_labels,
        montage_origin,
    )
    from detail.api.clip_views import blind_aliases

    if out_path is None and gif_dir is None and csv_path is None:
        raise click.UsageError("give at least one of --out, --per-clip-gifs and --csv")
    if (
        (max_width is not None or fps is not None)
        and out_path is None
        and gif_dir is None
    ):
        raise click.UsageError(
            "--max-width and --fps shape a picture: give --out or --per-clip-gifs with them"
        )
    proj = _open_experiment(experiment_dir)
    run = proj.authored_run(run_id)
    if run is None or not hasattr(run, "typical_members"):
        raise click.ClickException(
            f"{run_id} is not a temporal-clustering run of this experiment"
        )
    try:
        layers = MontageLayers(
            raw=raw,
            mask=mask,
            keypoints=keypoints,
            mask_trail=mask_trail,
            keypoint_trail=keypoint_trail,
            trail_points=tuple(trail_points) or None,
            aligned=aligned,
        )

        def work(bar) -> list[str]:
            lines_out: list[str] = []
            with cluster_montage(
                proj,
                run,
                node,
                limit=limit,
                sort=sort,
                context_s=context_s,
                progress=bar,
            ) as (montage, clips):
                aliases = (
                    blind_aliases([c.recording_id for c in clips]) if blind else {}
                )
                labels = montage_labels(clips, blind=blind, aliases=aliases or None)
                stems = [clip_file_stem(c, aliases.get(c.recording_id)) for c in clips]
                if gif_dir is not None and existing_gifs(gif_dir, stems):
                    # Before anything is written, so a refusal leaves nothing half done.
                    raise UserInputError(
                        f"{gif_dir} already holds GIFs of these names "
                        f"({', '.join(existing_gifs(gif_dir, stems)[:3])}); choose another "
                        "folder for --per-clip-gifs, or remove them first"
                    )
                if out_path is not None:
                    written = montage.export(
                        out_path,
                        layers,
                        labels,
                        max_width=max_width,
                        fps=fps,
                        progress=bar,
                    )
                    lines_out.append(f"wrote {written}")
                if gif_dir is not None:
                    gifs = montage.export_clip_gifs(
                        gif_dir,
                        layers,
                        labels,
                        stems,
                        max_width=max_width or GIF_DEFAULT_WIDTH,
                        fps=fps or GIF_DEFAULT_FPS,
                        progress=bar,
                    )
                    lines_out.append(f"wrote {len(gifs)} GIFs to {gif_dir}")
                if csv_path is not None:
                    # The list names the clips themselves, never their context.
                    lister = (
                        ClusterMontage(proj.dir, clips, workers=1)
                        if context_s
                        else montage
                    )
                    try:
                        listed = lister.export_clip_list(
                            csv_path, montage_origin(run, node), progress=bar
                        )
                    finally:
                        if lister is not montage:
                            lister.close()
                    lines_out.append(f"wrote {listed}")
            return lines_out

        for line in _with_stage_progress(work):
            click.echo(line)
    except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
        raise click.ClickException(str(e))


@analysis.command("query-eval")
@_EXPERIMENT_OPT
@click.option(
    "--search", "search_recordings", multiple=True,
    help=_QE["source_recording_ids"].description,
)  # fmt: skip
@click.option(
    "--embed-run", "embed_run_id", required=True, help=_QE["embed_run_id"].description
)
@click.option(
    "--behavior", "behaviors", multiple=True, required=True,
    help=_QE["behaviors"].description,
)  # fmt: skip
@click.option(
    "--stream", "stream_name", default=None,
    help=_QE["stream_name"].description,
)  # fmt: skip
@click.option(
    "--track", "track_id", type=_int_range(_QE_DOC["track_id"]),
    default=_QE["track_id"].default, show_default=True, help=_QE["track_id"].description,
)  # fmt: skip
@click.option(
    "--fusion", type=click.Choice(list(get_args(_QE["fusion"].annotation))),
    default=_QE["fusion"].default, show_default=True, help=_QE["fusion"].description,
)  # fmt: skip
@click.option(
    "--method", type=click.Choice(list(get_args(_QE["method"].annotation))),
    default=_QE["method"].default, show_default=True, help=_QE["method"].description,
)  # fmt: skip
@click.option(
    "--keyframe-select", "keyframe_select",
    type=click.Choice(list(get_args(_QE["keyframe_select"].annotation))),
    default=_QE["keyframe_select"].default, show_default=True,
    help=_QE["keyframe_select"].description,
)  # fmt: skip
@click.option(
    "--n-keyframes", "n_keyframes", type=_int_range(_QE_DOC["n_keyframes"]),
    default=_QE["n_keyframes"].default, show_default=True,
    help=_QE["n_keyframes"].description,
)  # fmt: skip
@click.option(
    "--band", type=FINITE_FLOAT, default=_QE["band_s"].default, show_default=True,
    help=_QE["band_s"].description,
)  # fmt: skip
@click.option(
    "--k", "k", type=_int_range(_QE_DOC["k"]),
    default=_QE["k"].default, show_default=True, help=_QE["k"].description,
)  # fmt: skip
@click.option(
    "--tau", "tau", type=_float_range(_QE_DOC["tau"]),
    default=_QE["tau"].default, show_default=True, help=_QE["tau"].description,
)  # fmt: skip
@click.option(
    "--n-negatives", "n_negatives", type=_int_range(_QE_DOC["n_negatives"]),
    default=_QE["n_negatives"].default, show_default=True,
    help=_QE["n_negatives"].description,
)  # fmt: skip
@click.option(
    # ``--max-gap`` is this option's original name, kept so existing sweep scripts run.
    "--split-gap",
    "--max-gap",
    "trajectory_split_gap_s",
    type=FINITE_FLOAT,
    default=None,
    help=_QE["trajectory_split_gap_s"].description,
)
@click.option(
    "--max-results", "max_results", type=_int_range(_QE_DOC["max_results"]),
    default=None, help=_QE["max_results"].description,
)  # fmt: skip
@click.option(
    "--n-exemplars", "n_exemplars", type=_int_range(_QE_DOC["n_exemplars"]),
    default=_QE["n_exemplars"].default, show_default=True, help=_QE["n_exemplars"].description,
)  # fmt: skip
@click.option(
    "--negative-pool", "negative_pool",
    type=click.Choice(list(get_args(_QE["negative_pool"].annotation))),
    default=_QE["negative_pool"].default, show_default=True,
    help=_QE["negative_pool"].description,
)  # fmt: skip
@click.option(
    "--n-holdout", "n_holdout", type=_int_range(_QE_DOC["n_holdout"]),
    default=None, help=_QE["n_holdout"].description,
)  # fmt: skip
@click.option(
    "--folds", type=_int_range(_QE_DOC["folds"]), default=_QE["folds"].default,
    show_default=True, help=_QE["folds"].description,
)  # fmt: skip
@click.option(
    "--seed", type=_int_range(_QE_DOC["seed"]), default=_QE["seed"].default,
    show_default=True, help=_QE["seed"].description,
)  # fmt: skip
@click.option(
    "--iou", "iou_threshold", type=FINITE_FLOAT, default=_QE["iou_threshold"].default,
    show_default=True, help=_QE["iou_threshold"].description,
)  # fmt: skip
@click.option(
    "--point-window", "point_window_s", type=FINITE_FLOAT, default=_QE["point_window_s"].default,
    show_default=True, help=_QE["point_window_s"].description,
)  # fmt: skip
@click.option(
    "--out", "out_path", required=True, type=UserPath(dir_okay=False, path_type=Path),
    help="Metrics CSV to append one row per behavior to (created if absent).",
)  # fmt: skip
@click.option(
    "--pr-out", "pr_out_dir", default=None, type=UserPath(file_okay=False, path_type=Path),
    help="Optional directory for per-behavior PR-curve points.",
)  # fmt: skip
@click.option(
    "--predictions-out", "predictions_out", default=None,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Optional CSV of per-clip prediction outcomes (TP/FP + matched GT). Overwritten.",
)  # fmt: skip
@click.option(
    "--ground-truth-out", "ground_truth_out", default=None,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Optional CSV of ground-truth outcomes (matched / FN / exemplar). Overwritten.",
)  # fmt: skip
@click.option(
    "--folds-out", "folds_out", default=None,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Optional CSV of per-fold metrics behind the mean±std row. Overwritten.",
)  # fmt: skip
@click.option(
    "--overlap-out", "overlap_out", default=None,
    type=UserPath(dir_okay=False, path_type=Path),
    help="Optional CSV of recall/precision/AP at each overlap level, each beside its "
         "null (a value that doesn't clear its null isn't real). Overwritten.",
)  # fmt: skip
def analysis_query_eval(
    experiment_dir, search_recordings, embed_run_id, behaviors, stream_name, track_id,
    fusion, method, keyframe_select, n_keyframes, band, k, tau, n_negatives, negative_pool,
    trajectory_split_gap_s, max_results, n_exemplars, n_holdout, folds, seed,
    iou_threshold, point_window_s, out_path, pr_out_dir, predictions_out, ground_truth_out,
    folds_out, overlap_out,
) -> None:  # fmt: skip
    """Score behavior-query retrieval against ground truth (§16, scriptable).

    Held-out exemplars seed the query; matches are scored against the remaining
    ground-truth intervals. One row per --behavior is appended to --out so a shell
    sweep over configs / embedding runs builds a comparison table."""
    from detail.pipeline.query_eval import (
        QueryEvalModule,
        _overlap_rows,
        evaluate_query,
        null_at as query_eval_null_at,
    )
    from detail.storage import query_eval_csv

    _open_experiment(
        experiment_dir
    )  # friendly error if the experiment is missing/invalid
    if out_path.is_file():
        # Rows are appended to --out after the whole evaluation; an unrelated CSV there
        # would refuse them only then, so check it before spending the evaluation.
        try:
            query_eval_csv.read_metrics_csv(out_path)
        except ValueError as e:
            raise click.ClickException(
                f"--out {out_path} is not a query-eval metrics CSV ({e}); rows are "
                "appended to it, so name a new file or an earlier --out"
            )
    config = {
        "source_recording_ids": list(search_recordings),
        "embed_run_id": embed_run_id,
        "behaviors": list(behaviors),
        "stream_name": stream_name,
        "track_id": track_id,
        "fusion": fusion,
        "method": method,
        "keyframe_select": keyframe_select,
        "n_keyframes": n_keyframes,
        "band_s": band,
        "k": k,
        "tau": tau,
        "n_negatives": n_negatives,
        "negative_pool": negative_pool,
        "trajectory_split_gap_s": trajectory_split_gap_s,
        "max_results": max_results,
        "n_exemplars": n_exemplars,
        "n_holdout": n_holdout,
        "folds": folds,
        "seed": seed,
        "iou_threshold": iou_threshold,
        "point_window_s": point_window_s,
    }
    # The module's own checks, which a launched run gets and this direct call skipped: an
    # unknown --stream, --embed-run, --track or --behavior fails naming what there is, rather
    # than scoring nothing (`docs/cli.md`).
    try:
        config = QueryEvalModule().validate(experiment_dir, config)
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    bar = _StageProgress()
    try:
        results = evaluate_query(experiment_dir, config, progress=bar)
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()

    rows = []
    fold_rows = []
    for r in results:
        if r.metrics is None:
            click.echo(f"{r.behavior} [{r.behavior_type}]: skipped — {r.skipped}")
            continue
        m = r.metrics
        spread = f" ±{m.ap_std:.3f}" if m.n_folds > 1 else ""
        click.echo(
            f"{r.behavior} [{r.behavior_type}]: AP={m.ap:.3f}{spread} "
            f"P={m.precision:.3f} R={m.recall:.3f} ({m.n_folds} fold(s), "
            f"tp={m.tp} fp={m.fp} fn={m.fn})"
        )
        at = query_eval_null_at(r.overlap, iou_threshold)
        rows.append(
            query_eval_csv.MetricRow(
                embed_run_id=embed_run_id, behavior=r.behavior,
                behavior_type=r.behavior_type, n_recordings=r.n_recordings, fusion=fusion,
                method=method,
                band_s=band, iou_threshold=iou_threshold, point_window_s=point_window_s,
                n_exemplars=n_exemplars, n_holdout=n_holdout, seed=seed,
                negative_pool=negative_pool,
                trajectory_split_gap_s=trajectory_split_gap_s,
                max_results=max_results, n_gt=m.n_gt, n_pred=m.n_pred, tp=m.tp, fp=m.fp,
                fn=m.fn, ap=m.ap, precision=m.precision, recall=m.recall,
                best_f1=m.best_f1, best_f1_threshold=m.best_f1_threshold,
                n_folds=m.n_folds, ap_std=m.ap_std, precision_std=m.precision_std,
                recall_std=m.recall_std, best_f1_std=m.best_f1_std,
                recall_null=at.recall_null if at else 0.0,
                recall_lift=at.recall_lift if at else 0.0,
                ap_null=at.ap_null if at else 0.0,
                ap_lift=at.ap_lift if at else 0.0,
            )
        )  # fmt: skip
        for j, fm in enumerate(r.fold_metrics):
            fold_rows.append(
                query_eval_csv.FoldMetricRow(
                    behavior=r.behavior, fold=j, n_gt=fm.n_gt, n_pred=fm.n_pred,
                    tp=fm.tp, fp=fm.fp, fn=fm.fn, ap=fm.ap, precision=fm.precision,
                    recall=fm.recall, best_f1=fm.best_f1,
                    best_f1_threshold=fm.best_f1_threshold,
                )
            )  # fmt: skip
            if pr_out_dir is not None:
                Path(pr_out_dir).mkdir(parents=True, exist_ok=True)
                query_eval_csv.write_pr_points(
                    Path(pr_out_dir) / f"{embed_run_id}_{r.behavior}_fold_{j:02d}.csv",
                    fm.pr_points,
                )
    if not rows:
        raise click.ClickException(
            "every behavior was skipped, so nothing was scored and --out was not written"
        )
    query_eval_csv.append_metrics(out_path, rows)
    click.echo(f"metrics: {out_path}")
    if folds_out is not None:
        query_eval_csv.write_folds_csv(folds_out, fold_rows)
        click.echo(f"folds: {folds_out}")
    if overlap_out is not None:
        query_eval_csv.write_overlap_csv(
            overlap_out, [row for r in results for row in _overlap_rows(r)]
        )
        click.echo(f"overlap: {overlap_out}")
    # Per-clip detail (the recorded run writes these into its run dir; the scriptable
    # path emits them on request, for surface parity with the GUI explorer).
    if predictions_out is not None:
        query_eval_csv.write_predictions_csv(
            predictions_out, [row for r in results for row in r.predictions]
        )
        click.echo(f"predictions: {predictions_out}")
    if ground_truth_out is not None:
        query_eval_csv.write_ground_truth_csv(
            ground_truth_out, [row for r in results for row in r.ground_truth]
        )
        click.echo(f"ground truth: {ground_truth_out}")


# --------------------------------------------------------------------------- #
# modules (processing-module registry, §6.5)
# --------------------------------------------------------------------------- #
@main.command("modules")
def modules() -> None:
    """List the registered processing modules and their dependency chain (§6.5)."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    if not PROCESSING_MODULES:
        click.echo("(no processing modules registered)")
        return
    click.echo("processing modules:")
    for mid, m in sorted(PROCESSING_MODULES.items()):
        click.echo(f"  {mid:14s} upstream={str(m.upstream):11s} runs_key={m.runs_key}")


_DOCTOR_GLYPH = {
    "ok": "✓",
    "warn": "•",
    "fail": "✗",
    "unknown": "?",
    "skipped": "–",
}


@main.command("doctor")
def doctor() -> None:
    """Check this machine's setup (§1): PyTorch/CUDA, the SAM 3 backend, Hugging Face, and
    model weights. Runs the checks in a fresh subprocess, so it reflects the *current* install
    state (handy for debugging an install over SSH). Each failing check names the docs page
    that explains the fix and prints the copy-paste command that fixes it. Exits non-zero if
    any check failed.
    """
    from detail.api import checks as api_checks

    results = {r.check_id: r for r in api_checks.run_checks()}
    groups: dict[str, list] = {}
    for info in api_checks.list_checks():
        groups.setdefault(info.group, []).append(info)

    failed = False
    needs_install = False
    setup_checks = {"torch", "sam3_backend", "triton", "dinov3_backend"}
    for group in sorted(groups):
        click.echo(f"\n{group}")
        for info in groups[group]:
            result = results.get(info.check_id)
            status = result.status.value if result is not None else "unknown"
            glyph = _DOCTOR_GLYPH.get(status, "?")
            detail = (
                f": {result.detail}" if result is not None and result.detail else ""
            )
            click.echo(f"  {glyph} {info.label}{detail}")
            if status == "fail":
                failed = True
            if status in ("fail", "warn"):
                if info.fix_page:
                    click.echo(f"      fix: detail docs {info.fix_page}")
                if info.check_id in setup_checks:
                    needs_install = True
    if needs_install:
        # The full install sequence for this machine — open a terminal to done. SAM 3
        # needs an NVIDIA GPU, so flag when this machine can't actually run it.
        cuda = results.get("cuda")
        gpu_note = (
            ""
            if cuda is not None and cuda.status.value == "ok"
            else " — SAM 3 needs an NVIDIA GPU to run"
        )
        script = api_checks.current_script()
        click.echo(f"\nInstall commands ({script.os_label}, this machine){gpu_note}:")
        for step in script.steps:
            click.echo(f"  # {step.note}")
            click.echo(f"  {step.command}")
    if failed:
        raise SystemExit(1)


@main.command("version")
@click.option(
    "--history",
    "n_history",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Also list the newest N changes on this checkout's branch.",
)
@click.option(
    "--check",
    is_flag=True,
    help=f"Fetch {buildinfo.UPSTREAM_REMOTE}/{buildinfo.UPSTREAM_BRANCH} and say how far "
    "behind this copy is (uses the network).",
)
def version(n_history: int, check: bool) -> None:
    """Which build of detail this is, what changed, and whether it is up to date.

    The headless twin of the GUI's Version History panel. Everything is read from the git
    checkout detail was installed from.
    """
    click.echo(buildinfo.build_info().describe())
    for entry in buildinfo.history(n_history) if n_history else []:
        click.echo(f"  {entry.date}  {entry.commit}  {entry.subject}")
    if check:
        # A fetch can take seconds; say what it is waiting on rather than sit silent.
        click.echo(
            f"Fetching {buildinfo.UPSTREAM_REMOTE}/{buildinfo.UPSTREAM_BRANCH}…",
            err=True,
        )
        try:
            status = buildinfo.check_for_updates()
        except buildinfo.UpdateCheckError as e:
            raise click.ClickException(str(e))
        click.echo(status.describe())


# --------------------------------------------------------------------------- #
# hf-token (machine-local credential)
# --------------------------------------------------------------------------- #
@main.command("hf-token")
@click.argument("token")
def hf_token(token: str) -> None:
    """Validate + store a Hugging Face token in ~/.config/detail/credentials.yaml."""
    from detail.storage import credentials

    try:
        account = credentials.set_hf_token(token)
    except Exception as e:  # huggingface_hub raises various subtypes
        raise click.ClickException(f"token validation failed: {e}")
    click.echo(f"signed in as: {account}")


@main.group("video-root")
def video_root_group() -> None:
    """Register the machine-local roots that reference-in-place videos live under."""


@video_root_group.command("set")
@click.argument("alias")
@click.argument("path", type=UserPath(exists=True, file_okay=False, path_type=Path))
def video_root_set(alias: str, path: Path) -> None:
    """Register (or re-point) video root ALIAS to directory PATH on this machine.

    Re-pointing an alias relinks every recording that references videos under it — the
    single-command fix for a whole video library that moved.
    """
    from detail.storage import video_roots

    try:
        video_roots.register_video_root(alias, path)
    except (ValueError, NotADirectoryError) as e:
        raise click.ClickException(str(e))
    click.echo(f"video root {alias!r} -> {path}")


@video_root_group.command("list")
def video_root_list() -> None:
    """List the video roots registered on this machine."""
    from detail.storage import video_roots

    roots = video_roots.list_video_roots()
    if not roots:
        click.echo("no video roots registered")
        return
    for alias, path in sorted(roots.items()):
        click.echo(f"{alias}\t{path}")


# --------------------------------------------------------------------------- #
# cache (§3) — this machine's display-only video previews
# --------------------------------------------------------------------------- #
@main.group()
def cache() -> None:
    """This machine's caches: the display-only video previews."""


@cache.group("previews")
def cache_previews() -> None:
    """Display-only video previews (small copies the clip montage plays from)."""


@cache_previews.command("build")
@_EXPERIMENT_OPT
@click.option(
    "--recording",
    "recording_ids",
    multiple=True,
    type=str,
    help="Recording name (or id) to build for (repeatable; default: every recording).",
)
def cache_previews_build(experiment_dir: Path, recording_ids: tuple[str, ...]) -> None:
    """Build every missing preview for an experiment's recordings on this machine.

    Each part's proxy, and each complete tracking run's masks. Already-built previews are
    skipped, so this is safe to re-run; minutes per hour of video.
    """
    proj = _open_experiment(experiment_dir)
    ids = [_recording(proj, ref).id for ref in recording_ids] or None
    bar = _StageProgress()
    try:
        report = proj.build_previews(ids, progress=bar)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    _report_preview_failures(proj, report)
    if report.built == 0 and report.ok:
        click.echo("previews: already built, nothing to do")
        return
    click.echo(f"previews: built {report.built}")
    if not report.ok:
        raise click.ClickException(f"{len(report.failed)} preview build(s) failed")


def _report_preview_failures(proj: Experiment, report) -> None:
    """One stderr line per failed preview build: the batch goes on past a failure, so each
    has to be said, not just counted."""
    names = proj.recording_names() if report.failed else {}
    for task, why in report.failed:
        click.echo(
            f"preview failed: {names.get(task.recording_id, task.recording_id)} "
            f"part {task.part_index + 1} {task.kind}: {why}",
            err=True,
        )


@cache_previews.command("size")
def cache_previews_size() -> None:
    """Print where this machine's previews are and how much disk they use."""
    from detail.api import previews
    from detail.api.exporters import format_size

    click.echo(
        f"{previews.preview_cache_dir()}\t"
        f"{format_size(previews.preview_cache_size())}"
    )


@cache.command("clear")
@click.option(
    "--previews",
    "clear_previews",
    is_flag=True,
    help="Delete every video preview on this machine (they rebuild when next needed).",
)
def cache_clear(clear_previews: bool) -> None:
    """Delete a cache on this machine. Experiment data is never touched."""
    from detail.api import previews
    from detail.api.exporters import format_size

    if not clear_previews:
        raise click.UsageError("say which cache to clear: --previews")
    freed = previews.clear_preview_cache()
    click.echo(f"cleared previews: {format_size(freed)} freed")


# --------------------------------------------------------------------------- #
# measures (§23) — a behavior result turned into numbers
# --------------------------------------------------------------------------- #
@main.group()
def measures() -> None:
    """Turn a behavior result into numbers: counts, total time, rates, durations."""


@measures.command("list")
def measures_list() -> None:
    """Every measure, with the unit its value carries and the window rule it follows."""
    from detail.analysis.measures import MEASURES

    width = max(len(name) for name in MEASURES)
    for name, spec in sorted(MEASURES.items()):
        needs = " needs observed time" if spec.needs_denominator else ""
        click.echo(
            f"{name:{width}}  {spec.label:26}  {spec.unit or '-':8}  "
            f"by {spec.rule}{needs}"
        )
        click.echo(f"{'':{width}}  {spec.description}")


@measures.command("export")
@_EXPERIMENT_OPT
@click.option("--run", "run_id", required=True, help="Analysis run id (e.g. an_01).")
@click.option(
    "--out", "out_dir", required=True, type=UserPath(file_okay=False, path_type=Path),
    help=(
        "Directory to create the export folder in; each export makes a new one holding "
        "measures.csv, frames.csv and measures.yaml."
    ),
)  # fmt: skip
@click.option(
    "--measure", "measures_opt", multiple=True,
    help="Measure to compute (repeatable; default: all of them, see `measures list`).",
)  # fmt: skip
@_window_options
def measures_export(
    experiment_dir, run_id, out_dir, measures_opt, windows, every_s, parts, length_s, step_s
) -> None:  # fmt: skip
    """Measure one analysis run's behavior output and write the tables.

    Any module that produces labeled behavior in time can be measured — behavior scan, temporal
    clustering — because they all write the same artifact.
    """
    proj = _open_experiment(experiment_dir)
    bar = _StageProgress()
    try:
        out = proj.measures(run_id).export(
            out_dir,
            windows=WindowSpec(
                **_window_spec(windows, every_s, parts, length_s, step_s)
            ),
            measures=list(measures_opt) or None,
            progress=bar,
        )
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    click.echo(f"wrote {out}")


def _refuse_unanswered_unit(experiment, group_by: list, recordings: list) -> None:
    """Raise unless the cohort has no repeats — and name the column when it does.

    The GUI infers the same column and pre-selects it (`analysis/cohort.py`); the CLI cannot
    pre-select anything, so it names it in the refusal. Surface parity is the point: a user who
    reads "pass a column" and does not know *which* column is left where the GUI's first version
    left everyone.
    """
    from detail.analysis import cohort

    metadata = experiment.recording_metadata
    ids = recordings or [r.id for r in experiment.recordings]
    subject = cohort.infer_subject_column(metadata, ids, group_by=group_by)
    if subject is None:
        return  # nothing repeats: `[]` is the only answer, and it is the one taken
    sizes = cohort.describe_columns(metadata, ids, [subject])[0]
    raise click.UsageError(
        f"--aggregate-to is required here: {sizes.n_values} value(s) of '{subject}' cover these "
        f"{len(ids)} recordings, one of them {max(sizes.counts.values())} times, so pooling and "
        f"averaging-within-{subject} answer different questions. Pass --aggregate-to {subject} "
        f"to give each one a single vote, or --aggregate-to none to count every recording "
        f"separately."
    )


def _comparison_default(name: str):
    """A `comparison run` option's default, read off the config model rather than retyped."""
    from detail.configs.comparison import ComparisonConfig

    return ComparisonConfig.model_fields[name].default


def _comparison_help(name: str) -> str:
    """A `comparison run` option's help, read off the config model rather than retyped."""
    from detail.configs.comparison import ComparisonConfig

    return ComparisonConfig.model_fields[name].description or ""


@main.group()
def comparison() -> None:
    """Compare behavior between groups defined by recording metadata (§24)."""


@comparison.command("run")
@_EXPERIMENT_OPT
@click.option(
    "--source-run",
    "source_run_id",
    required=True,
    help="Analysis run holding the behavior output.",
)
@click.option(
    "--snapshot",
    "source_snapshot",
    default=None,
    help="Pinned copy of that run's artifact to compare (snap_NN). Omit and the run's current "
    "artifact is pinned and the id recorded, so the numbers stay reproducible after a re-cut.",
)
@click.option(
    "--group-by",
    "group_by",
    multiple=True,
    required=True,
    help="Metadata column to compare across (repeatable); also 'subject' or 'window_id'.",
)
@click.option(
    "--aggregate-to",
    "aggregate_to",
    multiple=True,
    help="Column to average through before grouping — the unit of analysis. "
    "Pass --aggregate-to none for one observation per recording. REQUIRED.",
)
@click.option(
    "--source",
    type=click.Choice(["labels", "nodes"]),
    default=_comparison_default("source"),
    show_default=True,
    help=_comparison_help("source"),
)
@click.option(
    "--level",
    "levels",
    multiple=True,
    help="A value of the --group-by column; pass twice to contrast the second against the "
    "first. " + _comparison_help("levels"),
)
@click.option(
    "--recording", "recordings", multiple=True, help="Restrict to these recordings."
)
@click.option(
    "--measure", "measures_opt", multiple=True, help="Measure to compute (repeatable)."
)
@_window_options
@click.option(
    "--draws",
    type=int,
    default=_comparison_default("draws"),
    show_default=True,
    help=_comparison_help("draws"),
)
@click.option(
    "--seed",
    type=int,
    default=_comparison_default("seed"),
    show_default=True,
    help=_comparison_help("seed"),
)
@click.option(
    "--presence-s",
    type=FINITE_FLOAT,
    default=_comparison_default("presence_s"),
    show_default=True,
    help=_comparison_help("presence_s"),
)
@click.option(
    "--pseudo-count-s",
    type=FINITE_FLOAT,
    default=_comparison_default("pseudo_count_s"),
    show_default=True,
    help=_comparison_help("pseudo_count_s"),
)
@click.option(
    "--min-units-fraction",
    type=FINITE_FLOAT,
    default=_comparison_default("min_units_fraction"),
    show_default=True,
    help=_comparison_help("min_units_fraction"),
)
@click.option(
    "--candidate-floor-s",
    type=FINITE_FLOAT,
    default=_comparison_default("candidate_floor_s"),
    show_default=True,
    help=_comparison_help("candidate_floor_s"),
)
@click.option(
    "--suppress-nesting/--no-suppress-nesting",
    default=_comparison_default("suppress_nesting"),
    show_default=True,
    help=_comparison_help("suppress_nesting"),
)
@click.option(
    "--statistic",
    type=click.Choice(["moderated_t", "effect"]),
    default=_comparison_default("statistic"),
    show_default=True,
    help=_comparison_help("statistic"),
)
def comparison_run(
    experiment_dir,
    source_run_id,
    source_snapshot,
    group_by,
    aggregate_to,
    source,
    levels,
    recordings,
    measures_opt,
    windows,
    every_s,
    parts,
    length_s,
    step_s,
    draws,
    seed,
    presence_s,
    pseudo_count_s,
    min_units_fraction,
    candidate_floor_s,
    suppress_nesting,
    statistic,
) -> None:
    """Run a comparison and save it.

    ``--aggregate-to`` says what one observation is. It is required *when the cohort has
    repeats* — when some animal contributed more than one of the recordings being compared,
    because then pooling and averaging-within-animal answer different questions and the tool must
    not choose (§24). When nobody contributed two there is nothing to decide, and demanding an
    answer is a question with one possible response.

    The design — between units, paired within them, a trend, or a profile across three or more
    named values — follows from what the ``--group-by`` column does within the unit, and is
    printed when the run ends (§24).
    ``--source nodes`` screens every node of a clustering run's node cache, named or not.
    """
    proj = _open_experiment(experiment_dir)
    if not aggregate_to:
        _refuse_unanswered_unit(proj, list(group_by or ()), list(recordings or ()))
    path = [] if tuple(aggregate_to) == ("none",) else list(aggregate_to)
    if levels and len(levels) != 2:
        raise click.UsageError(
            f"--level names the two values to contrast, so pass it exactly twice "
            f"(got {len(levels)})."
        )
    from detail.api.comparison import run_comparison

    bar = _StageProgress()
    try:
        comp = run_comparison(
            proj,
            {
                "source_run_id": source_run_id,
                "source_snapshot": source_snapshot,
                "source": source,
                "group_by": list(group_by),
                "aggregate_to": path,
                "levels": list(levels) or None,
                "recordings": list(recordings) or None,
                "measures": list(measures_opt) or None,
                "windows": _window_spec(windows, every_s, parts, length_s, step_s),
                "draws": draws,
                "seed": seed,
                "presence_s": presence_s,
                "pseudo_count_s": pseudo_count_s,
                "min_units_fraction": min_units_fraction,
                "candidate_floor_s": candidate_floor_s,
                "suppress_nesting": suppress_nesting,
                "statistic": statistic,
            },
            progress=bar,
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    record = comp.record()
    click.echo(
        f"{comp.id}: {record['n_observations']} observation(s) over {len(record['groups'])} group(s)"
    )
    click.echo(f"  one observation is: {'|'.join(path) or '(one per recording)'}")
    click.echo(f"  design: {record.get('design')} — {record.get('design_reason', '')}")
    from detail.analysis import comparison_report

    if comparison_report.statistic_note(record):
        click.echo(f"  {comparison_report.statistic_note(record)}")
    if comparison_report.left_out_sentence(record):
        click.echo(
            f"  left out: "
            f"{comparison_report.left_out_sentence(record, proj.recording_names())}"
        )
    if record.get("design") != "between":
        click.echo(
            f"  an effect is undefined below {record.get('min_units')} present unit(s) of "
            f"{record.get('n_units')}"
        )
    if record.get("source") == "nodes":
        click.echo(
            f"  every node of {record['source_run_id']} at or above "
            f"{record.get('candidate_floor_s')} s, ranked on share of time"
        )
    if record.get("source_snapshot"):
        click.echo(f"  read {record['source_run_id']} @ {record['source_snapshot']}")
    _echo_screened(record)
    click.echo(f"  {comp.dir}")


def _echo_screened(record: dict) -> None:
    """The one line a labels comparison prints when a difference screen over the same column
    chose a label it scores (``comparison_report.screened_sentence``)."""
    from detail.analysis import comparison_report

    sentence = comparison_report.screened_sentence(record)
    if sentence:
        click.echo(f"  not independent: {sentence}")


def _figure_themes() -> list[str]:
    """The chart themes `--theme` offers, read from `chart_theme` rather than retyped here."""
    from detail.core.chart_theme import THEMES

    return sorted(THEMES)


_THEME_HELP = (
    "Background the figures are drawn on: light (white, for print; the default) or dark "
    "(for slides). The same design either way."
)


@comparison.command("export")
@click.argument("comparison_id")
@_EXPERIMENT_OPT
@click.option(
    "--out",
    "out_dir",
    type=UserPath(path_type=Path),
    default=None,
    help="Where to write; defaults into the comparison's own folder.",
)
@click.option(
    "--format",
    "formats",
    multiple=True,
    default=("pdf", "svg", "png"),
    show_default=True,
    help="A figure format to write; repeat for several. Any format matplotlib writes "
    "(jpg, tiff, eps, …); an unknown one is refused before anything is drawn.",
)
@click.option(
    "--theme",
    "theme",
    type=click.Choice(_figure_themes()),
    default="light",
    show_default=True,
    help=_THEME_HELP,
)
def comparison_export(comparison_id, experiment_dir, out_dir, formats, theme) -> None:
    """Write a comparison's tables and figures. Re-runnable; no recompute."""
    proj = _open_experiment(experiment_dir)
    comp = proj.get_comparison(comparison_id)
    changed = comp.drift()
    if changed:
        click.echo("warning: inputs changed since this comparison ran:", err=True)
        for line in changed:
            click.echo(f"  {line}", err=True)
    bar = _StageProgress()
    try:
        manifest = comp.export(
            out_dir, formats=tuple(formats), theme=theme, progress=bar
        )
    finally:
        bar.close()
    click.echo(f"{len(manifest['artifacts'])} artifact(s) -> {out_dir or comp.dir}")


@comparison.command("list")
@_EXPERIMENT_OPT
def comparison_list(experiment_dir) -> None:
    """List the experiment's comparisons (cmp_NN)."""
    proj = _open_experiment(experiment_dir)
    comps = proj.comparisons
    if not comps:
        click.echo("(no comparisons)")
        return
    for comp in comps:
        record = comp.record()
        click.echo(
            f"{comp.id}  {','.join(record.get('group_by') or [])}"
            f"  by={'|'.join(record.get('aggregate_to') or []) or 'recording'}"
            f"  n={record.get('n_observations', 0)}"
            + (f"  design={record['design']}" if record.get("design") else "")
            + ("  source=nodes" if record.get("source") == "nodes" else "")
            + (
                f"  screened={','.join(sorted(record['screened_labels']))}"
                if record.get("screened_labels")
                else ""
            )
        )


@comparison.command("delete")
@click.argument("comparison_id")
@_EXPERIMENT_OPT
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def comparison_delete(comparison_id, experiment_dir, yes) -> None:
    """Delete a comparison and everything it wrote, exports included."""
    proj = _open_experiment(experiment_dir)
    if not yes:
        click.confirm(
            f"Delete {comparison_id} and its exported tables and figures?", abort=True
        )
    try:
        proj.delete_comparison(comparison_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"deleted {comparison_id}")


@main.group()
def tools() -> None:
    """Tools (§27) — one-off runs that need no experiment."""


@tools.command("quick-segment")
@click.argument(
    "videos", nargs=-1, required=True,
    type=UserPath(exists=True, dir_okay=False, path_type=Path),
)  # fmt: skip
@click.option(
    "--prompt",
    "prompts",
    multiple=True,
    required=True,
    help="A prompt to try (repeatable). Each repeat is a SEPARATE prompt, compared "
    "side by side in sample mode; put several classes in one prompt by "
    "comma-separating them ('mouse,rat').",
)
# `mode` as two flags sharing one destination, so the option *is* the model's field: its
# default and its help are read from QuickSegmentConfig rather than typed here.
@click.option(
    "--sample",
    "mode",
    flag_value="sample",
    default=_QSEG_DOC["mode"]["default"],
    show_default=True,
    help=_QSEG_DOC["mode"]["help"],
)
@click.option(
    "--consecutive",
    "mode",
    flag_value="consecutive",
    help="Select the consecutive mode described under --sample.",
)
@click.option(
    "--whole-video",
    "whole_video",
    is_flag=True,
    default=_QSEG_DOC["whole_video"]["default"],
    help=_QSEG_DOC["whole_video"]["help"],
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="Detection model id. Default: auto (SAM3 on an NVIDIA GPU).",
)
@click.option(
    "--frames",
    "n_frames",
    type=_int_range(_QSEG_DOC["n_frames"]),
    default=_QSEG_DOC["n_frames"]["default"],
    show_default=True,
    help=_QSEG_DOC["n_frames"]["help"],
)
@_MIN_CONF_OPT
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default=None,
    help="Compute backend (default: the best one this machine has that the model "
    "supports: CUDA, then MPS, then CPU).",
)
@_DEVICE_OPT
@click.option(
    "--export",
    "export_path",
    type=UserPath(dir_okay=False, path_type=Path),
    default=None,
    help="Also render the overlay clip to this .mp4 (playback is GUI-only).",
)
def tools_quick_segment(
    videos: tuple[Path, ...],
    prompts: tuple[str, ...],
    mode: str,
    whole_video: bool,
    model_id: str | None,
    n_frames: int,
    min_conf: float,
    backend: str | None,
    devices: tuple[int, ...],
    export_path: Path | None,
) -> None:
    """Segment VIDEOS with one or more prompts and save a managed tool run."""
    from detail.api import models as api_models

    if model_id is None:
        try:
            model_id = api_tools.default_model_for_machine()
        except RuntimeError as e:
            raise click.ClickException(str(e))
    if backend is None:
        # The GUI's compute row picks the same way (gui/compute_target.ComputeTargetRow), and
        # the resolved backend is what the run's manifest records.
        backend = best_supported_backend(api_models.supported_backends(model_id))
    sample = mode == "sample"
    if not sample and len(videos) > 1:
        raise click.ClickException(
            "--consecutive reads one clip straight through; pass a single video "
            "(or use --sample to cover several)"
        )
    if not sample and len(prompts) > 1:
        raise click.ClickException(
            "--consecutive runs a single prompt; use --sample to compare prompts"
        )
    if sample and export_path is not None:
        # Checked before the run, which takes minutes: only a consecutive run has a clip.
        raise click.ClickException(
            "--export needs --consecutive: --sample frames are spread across the clip, so "
            "there is no clip to render"
        )
    parsed = [
        Prompt(
            targets=[
                PromptTarget(class_name=c.strip(), cue=TextCue(c.strip()))
                for c in text.split(",")
                if c.strip()
            ]
        )
        for text in prompts
    ]
    config = {
        "videos": [str(v) for v in videos],
        "video": str(videos[0]),  # singular alias for the run manifest
        "prompts": parsed,
        "prompt": parsed[0],
        "model_id": model_id,
        "model_variant": None,
        "compute": _compute_target(backend, devices),
        "mode": mode,
        "whole_video": whole_video,
        "n_frames": n_frames,
    }
    bar = _StageProgress()
    try:
        record = api_tools.run_tool("quick_segment", config, progress=bar)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    finally:
        bar.close()
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    click.echo(f"quick-segment {record['id']}: {record['result']['summary']}")
    _echo_quick_segment_kept(run_dir, record, min_conf)
    click.echo(f"saved to {run_dir}")
    if export_path is not None:
        from detail.tools.quick_segment import export_mp4

        ebar = _StageProgress()
        try:
            export_mp4(run_dir, record, export_path, progress=ebar, min_conf=min_conf)
        finally:
            ebar.close()
        click.echo(f"exported {export_path} at --min-conf {min_conf}")


@tools.command("list")
def tools_list() -> None:
    """List saved tool runs."""
    runs = api_tools.list_tool_runs()
    if not runs:
        click.echo("(no tool runs)")
        return
    for r in runs:
        summary = (r.get("result") or {}).get("summary") or r.get("status", "")
        click.echo(f"  {r['id']:9s} {r['tool_id']:16s} {r['status']:9s} {summary}")


@tools.command("delete")
@click.argument("run_id")
def tools_delete(run_id: str) -> None:
    """Delete a saved tool run by id."""
    try:
        api_tools.read_tool_run(run_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    api_tools.delete_tool_run(run_id)
    click.echo(f"deleted {run_id}")


@main.command("docs")
@click.option(
    "--serve",
    is_flag=True,
    help="Run a local live-reloading docs server (mkdocs serve) instead of opening "
    "the built site.",
)
@click.argument("topic", required=False)
def docs_cmd(serve: bool, topic: str | None) -> None:
    """Open the DETAIL documentation in your browser.

    Builds the site into a per-user cache on first use (offline; no network), then
    opens it. TOPIC is an optional page name — e.g. ``concepts``, ``cli``,
    ``reference/parameters``.
    """
    if serve:
        source = docs_access.docs_source_dir()
        subprocess.run(
            [sys.executable, "-m", "mkdocs", "serve", "-f", str(source / "mkdocs.yml")],
            cwd=str(source),
            check=True,
        )
        return
    bar = _StageProgress()
    try:
        docs_access.ensure_site_built(progress=bar)
    finally:
        bar.close()
    page = topic or "index"
    if topic and not (docs_access.built_site_dir() / f"{page}.html").is_file():
        raise click.ClickException(
            f"No documentation page '{topic}'. Try a nav path like 'concepts' or "
            "'reference/parameters'."
        )
    url = docs_access.help_url(page)
    click.echo(url)
    click.launch(url)


# --------------------------------------------------------------------------- #
# export (#336)
# --------------------------------------------------------------------------- #
_SUBJECT_OPT = click.option(
    "--subject",
    "subject_kind",
    type=click.Choice(["experiment", "recording", "run", "analysis_run", "distill"]),
    default=None,
    help="What to export from. Inferred from whichever of --recording / --run / "
    "--analysis-run / --distill-run you pass; required only when none is given "
    "(an experiment).",
)

#: Each id-carrying subject kind and the flag that names it.
_SUBJECT_FLAGS = {
    "recording": "--recording",
    "run": "--run",
    "analysis_run": "--analysis-run",
    "distill": "--distill-run",
}

_DISTILL_RUN_OPT = click.option(
    "--distill-run", "distill_run", help="Distill run id (distill_NNN)."
)


def _resolve_subject(
    proj, subject_kind, recording, run, analysis_run, distill_run=None
):
    """Build the ``ExportSubject`` from the id flags, refusing an ambiguous combination.

    Explicit rather than inferred from a bare id (OQ-EXP-B): inference reads better right up
    until a recording and a run share a name, and then it silently exports the wrong thing.
    Passing the id under its own flag *is* the subject kind, so the common case still costs
    one word. A recording is given by name or id, like every ``--recording``; an id that names
    nothing is refused here rather than listed as a subject every target is unavailable for.
    """
    from detail.api.exporters import ExportSubject
    from detail.api.exporters import check_subject

    experiment_dir = proj.dir

    given = {
        "recording": recording,
        "run": run,
        "analysis_run": analysis_run,
        "distill": distill_run,
    }
    named = {kind: value for kind, value in given.items() if value}
    if len(named) > 1:
        raise click.ClickException(
            "pass exactly one of "
            + " / ".join(_SUBJECT_FLAGS.values())
            + ", not "
            + ", ".join(_SUBJECT_FLAGS[k] for k in sorted(named))
        )
    if named:
        kind, value = next(iter(named.items()))
        if subject_kind and subject_kind != kind:
            raise click.ClickException(
                f"--subject {subject_kind} contradicts {_SUBJECT_FLAGS[kind]}"
            )
        if kind == "recording":
            value = _recording(proj, value).id
        subject = ExportSubject(kind, experiment_dir, value)
        check_subject(subject)
        return subject
    if subject_kind and subject_kind != "experiment":
        raise click.ClickException(
            f"--subject {subject_kind} needs an id: pass "
            f"{_SUBJECT_FLAGS[subject_kind]} <id>"
        )
    return ExportSubject("experiment", experiment_dir)


@main.group()
def export() -> None:
    """Get your data out — one catalogue of targets, the same on every surface (#336).

    Every target writes the data behind its artifact as well as the artifact, so a figure can be
    rebuilt outside DETAIL rather than only looked at. Each export folder also carries a
    ``params.csv`` (the settings that produced the run) and an ``export.yaml`` manifest.
    """


@export.command("list")
@_EXPERIMENT_OPT
@_SUBJECT_OPT
@click.option("--recording", help="Recording id to export from.")
@click.option(
    "--run", help="Experiment-level processing run id (run_NNN) to export from."
)
@click.option("--analysis-run", "analysis_run", help="Analysis run id (an_NN).")
@_DISTILL_RUN_OPT
def export_list(
    experiment_dir: Path,
    subject_kind: str | None,
    recording: str | None,
    run: str | None,
    analysis_run: str | None,
    distill_run: str | None,
) -> None:
    """Show what this subject can produce, and what ships beside each artifact."""
    from detail.api import exporters

    proj = _open_experiment(experiment_dir)
    subject = _resolve_subject(
        proj, subject_kind, recording, run, analysis_run, distill_run
    )
    targets = exporters.targets_for(subject)
    if not targets:
        click.echo(f"(nothing to export from this {subject.kind.replace('_', ' ')})")
        return
    for target in targets:
        flag = "  [portable]" if target.portable else ""
        click.echo(f"{target.id}{flag}")
        click.echo(f"    {target.label}: {target.artifact}")
        click.echo(f"    also writes: {', '.join(target.included_files())}")
        if target.scopes:
            click.echo(f"    scopes: {', '.join(target.scopes)}")
        # Every target this kind of subject *can* have is listed, available or not, so the
        # reason has to be here: a row with no explanation leaves the reader unable to tell
        # "DETAIL cannot export this" from "this run has none of it yet", and only the second
        # is something they can act on. The GUI has said it since #336; this never did.
        reason = exporters.unavailable_reason(target, subject)
        if reason is not None:
            click.echo(f"    cannot run: {reason}")


def _clip_renders() -> list[str]:
    """The clip flavours `--render` offers, read from `clip_render` rather than retyped here.

    A hand-written list is right on the day it is written and silently wrong the day a fourth
    flavour is added — the same reason `--method`'s choices come from the registry. Imported
    inside the function so the cost lands once at decoration, not on every name in this module.
    """
    from detail.api import clip_render

    return list(clip_render.RENDERS)


def _frame_range(text: str | None) -> tuple[int, int] | None:
    """``"1200:1340"`` → ``(1200, 1340)``, inclusive. ``None`` stays ``None``.

    Refused rather than guessed at: a range that silently became "everything" would export a
    forty-minute video when a ten-second clip was asked for, and the mistake is only visible
    once the file is written.
    """
    if text is None:
        return None
    start, sep, end = text.partition(":")
    if not sep or not start.strip().isdigit() or not end.strip().isdigit():
        raise click.BadParameter(
            f"--frames takes START:END, both whole frame numbers (got {text!r})"
        )
    return int(start), int(end)


@export.command("run")
@_EXPERIMENT_OPT
@_SUBJECT_OPT
@click.option("--recording", help="Recording id to export from.")
@click.option(
    "--run", help="Experiment-level processing run id (run_NNN) to export from."
)
@click.option("--analysis-run", "analysis_run", help="Analysis run id (an_NN).")
@_DISTILL_RUN_OPT
@click.option(
    "--target", "target_ids", multiple=True, required=True,
    help="Target id (see `export list`). Repeatable — several targets go into one folder.",
)  # fmt: skip
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=UserPath(file_okay=False, path_type=Path),
    help="Where to put the export folder. The folder itself is named for you.",
)
@click.option(
    "--cluster", default=None, help="Export only this cluster (a source label)."
)
@click.option(
    "--instance", "instance_ids", multiple=True,
    help="Export only these instances, by instance_id. Repeatable — this is how a selection "
         "is exported rather than a whole cluster.",
)  # fmt: skip
@click.option(
    "--frames", "frames", default=None, metavar="START:END",
    help="Export only this inclusive frame range, e.g. 1200:1340: the frames `--target clip` "
         "writes, or the rows of every file `--target frame_labels` writes.",
)  # fmt: skip
@click.option(
    "--track", "track_id", type=int, default=None,
    help="Which subject a clip outlines (default 0).",
)  # fmt: skip
@click.option(
    "--embed-run", "embed_run_id", default=None,
    help="The embedding run a clip's frames are read through; it resolves the tracking and "
         "detection runs the masks come from.",
)  # fmt: skip
@click.option(
    "--render", "renders", multiple=True,
    type=click.Choice(_clip_renders()),
    help="Clip flavour: raw footage, the mask drawn over it, or the mask alone. Repeatable — "
         "several flavours cost one decode, not several.",
)  # fmt: skip
@click.option(
    "--crop/--full-frame", "crop", default=False,
    help="Cut a square around the subject rather than keeping the whole field of view.",
)  # fmt: skip
@click.option(
    "--size", "size", type=int, default=None,
    help="Longest side of an exported clip, in pixels (default 640).",
)  # fmt: skip
@click.option(
    "--for-recording", "for_recordings", multiple=True,
    help="Export only this recording (an id or a name) of a run or analysis run. Repeatable. "
         "Without it every recording is exported, which for video is the size of the "
         "whole run: see `--estimate`. A target that writes one file for the "
         "whole run says so in `export list` (no `recordings` under its scopes).",
)  # fmt: skip
@click.option(
    "--estimate", "estimate_only", is_flag=True, default=False,
    help="Print how big the export would be and how long it would take, per target and in "
         "total, then stop without writing anything. A video target times a few dozen frames "
         "(about a second); the rest is read from what is on disk.",
)  # fmt: skip
@click.option(
    "--ignore-free-space", "ignore_free_space", is_flag=True, default=False,
    help="Start even when the estimate says the export will not fit on the destination's "
         "drive. Without it the command refuses, because a 100 GB export onto a nearly full "
         "drive fails an hour in.",
)  # fmt: skip
@click.option(
    "--theme", "theme", type=click.Choice(_figure_themes()), default=None,
    help=_THEME_HELP,
)  # fmt: skip
def export_run(
    experiment_dir: Path,
    subject_kind: str | None,
    recording: str | None,
    run: str | None,
    analysis_run: str | None,
    distill_run: str | None,
    target_ids: tuple[str, ...],
    out_dir: Path,
    cluster: str | None,
    instance_ids: tuple[str, ...],
    frames: str | None,
    track_id: int | None,
    embed_run_id: str | None,
    renders: tuple[str, ...],
    crop: bool,
    size: int | None,
    for_recordings: tuple[str, ...],
    estimate_only: bool,
    ignore_free_space: bool,
    theme: str | None,
) -> None:
    """Write the chosen targets into one new folder under ``--out``, as a run.

    The folder is named for you — experiment, module, run, and when — so two exports never
    collide and one is still identifiable months later on somebody else's machine.

    Blocks until complete. **Ctrl-C cancels it** — a foreground command you interrupt should
    stop, and somebody who sees a long estimate on a ``tracks`` export and reaches for Ctrl-C
    means it. To start one and walk away, launch it and come back with ``export attach``, or use
    the app, where an export never blocks anything.
    """
    from detail.api import exporters

    proj = _open_experiment(experiment_dir)
    subject = _resolve_subject(
        proj, subject_kind, recording, run, analysis_run, distill_run
    )
    try:
        # Ids, whichever of id or name was typed: the scope names recordings by id.
        recording_ids = tuple(proj.resolve_recording(ref).id for ref in for_recordings)
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(_error_text(e))
    scope = exporters.ExportScope(
        cluster=cluster,
        instance_ids=tuple(instance_ids),
        frames=_frame_range(frames),
        recording_ids=recording_ids,
    )
    # Only what was actually asked for. A default sent as an option is a value the user never
    # chose overriding a target's own default, and `**options` reaches every target in the run.
    options: dict = {}
    if renders:
        options["renders"] = list(
            renders
        )  # a list: this rides YAML, where a tuple is unsafe
        options["crop"] = crop
    if size is not None:
        options["size"] = size
    if track_id is not None:
        options["track_id"] = track_id
    if embed_run_id:
        options["embed_run_id"] = embed_run_id
    if theme is not None:
        options["theme"] = theme
    # What it will cost, before it is spent. Asked of the same catalogue the dialog asks, with
    # the same options, so the two surfaces cannot disagree about the same export.
    try:
        asked = proj.estimate_export(
            subject.kind, subject.id, list(target_ids), out_dir, scope=scope, **options
        )
    except (exporters.ExportError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    _echo_estimate(
        asked, detailed=estimate_only or len(target_ids) > 1, out_dir=out_dir
    )
    if estimate_only:
        return
    if asked.fits is False and not ignore_free_space:
        raise click.ClickException(
            f"this export is estimated at {exporters.format_size(asked.bytes)} but "
            f"{exporters.format_size(asked.free_bytes)} is free where {out_dir} is: choose "
            "another --out, narrow it with --for-recording, or pass --ignore-free-space"
        )
    try:
        handle = proj.run_export(
            subject.kind, subject.id, list(target_ids), out_dir, scope=scope, **options
        )
    except (exporters.ExportError, FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    click.echo(f"submitted export: {handle.id}")
    _stream_progress(handle, "export")
    _echo_export_result(proj, handle.id)


def _echo_estimate(asked, *, detailed: bool, out_dir: Path) -> None:
    """The estimate as the terminal says it: a total, then what it was measured against.

    ``detailed`` adds a line per target with how it was worked out, which is the thing to read
    when a number looks wrong. The space line is printed only when it matters or is asked for.
    """
    from detail.api import exporters

    click.echo(f"estimate: {exporters.describe_estimate(asked)}")
    if detailed:
        for target_id, estimate in asked.per_target:
            click.echo(
                f"  {target_id}: {exporters.describe_one(estimate)}"
                + (f"  ({estimate.basis})" if estimate.basis else "")
            )
    if asked.free_bytes is not None and (detailed or asked.fits is False):
        click.echo(
            f"  free on the drive for {out_dir}: {exporters.format_size(asked.free_bytes)}"
        )


def _echo_export_result(proj, run_id: str) -> None:
    """What the run wrote, from its record — the same lines the inline version printed.

    Read back rather than returned, because a durable run may finish in a worker after this
    process has stopped watching. A `partial` export names the targets that failed: it is the
    state that has something to say and the one a bare status word erases.
    """
    from detail.api import exporters

    record = proj.export_record(run_id)
    status = record.get("status", "")
    if record.get("empty"):
        click.echo(record["empty"])
        return
    files = record.get("files") or []
    click.echo(
        f"Wrote {len(files)} file(s) to {exporters.destination_of(record) or record.get('folder', '')}"
    )
    for name in files:
        click.echo(f"  {name}")
    failed = [o for o in (record.get("outcomes") or []) if o.get("status") == "failed"]
    for outcome in failed:
        click.echo(
            f"  {outcome['target']}: failed — {outcome.get('error', '')}", err=True
        )
    if status not in ("complete",):
        raise click.ClickException(
            f"export {run_id} finished {status!r}; see {record.get('error') or 'the run record'}"
        )


@export.command("attach")
@click.argument("run_id")
@_EXPERIMENT_OPT
def export_attach(run_id: str, experiment_dir: Path) -> None:
    """Reattach to an export run — stream a live one's progress (it keeps running with the
    app/CLI closed) or print a finished one's status. Ctrl-C detaches; it does not cancel the
    run (use ``export cancel`` for that)."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.export_record(run_id)  # ClickException rather than a traceback
    except FileNotFoundError:
        raise click.ClickException(f"no export run {run_id!r} in {experiment_dir}")
    handle = proj.attach_export_run(run_id)
    _stream_progress(handle, "export")
    click.echo(f"status: {handle.status}")


@export.command("cancel")
@click.argument("run_id")
@_EXPERIMENT_OPT
def export_cancel(run_id: str, experiment_dir: Path) -> None:
    """Stop an active export run. Whatever it has already written is kept — an export is
    stopped between targets, not killed mid-file. No-op once terminal."""
    proj = _open_experiment(experiment_dir)
    try:
        proj.cancel_export(run_id)
    except FileNotFoundError:
        raise click.ClickException(f"no export run {run_id!r} in {experiment_dir}")
    click.echo(f"cancelled export: {run_id}")


@main.group()
def notes() -> None:
    """Your own notes on an entry — the Notes box on every page of the GUI (§15).

    KIND is what the note is on, and ENTRY_ID which one (`experiment` takes no id). Notes are
    plain text kept beside the entry, as ``notes.md``, and travel with it into exports.
    """


def _notes_kinds() -> list[str]:
    """The entry kinds this build shows, read from `storage.notes` rather than retyped here.

    Every kind is surfaced today. A kind hidden later (§25) is left out of the choices, as its
    pages would be; the API still reads and writes its notes, because hiding governs launching.
    """
    from detail.storage import notes as notes_store

    return [kind for kind in notes_store.KINDS if surfaces.is_surfaced(kind)]


_NOTES_KIND_ARG = click.argument("kind", type=click.Choice(_notes_kinds()))
_NOTES_ID_ARG = click.argument("entry_id", required=False, default="")


@notes.command("show")
@_EXPERIMENT_OPT
@_NOTES_KIND_ARG
@_NOTES_ID_ARG
def notes_show(experiment_dir: Path, kind: str, entry_id: str) -> None:
    """Print the notes on one entry."""
    experiment = _open_experiment(experiment_dir)
    try:
        text = experiment.notes(kind, entry_id)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    click.echo(text if text else "(no notes)")


@notes.command("set")
@_EXPERIMENT_OPT
@_NOTES_KIND_ARG
@_NOTES_ID_ARG
@click.option("--text", help="The new notes, replacing what was there.")
@click.option(
    "--file",
    "source",
    type=UserPath(exists=True, dir_okay=False, allow_dash=True, path_type=Path),
    help="Read the new notes from this file (`-` for standard input).",
)
@click.option("--clear", is_flag=True, help="Remove the notes.")
def notes_set(
    experiment_dir: Path,
    kind: str,
    entry_id: str,
    text: str | None,
    source: Path | None,
    clear: bool,
) -> None:
    """Replace the notes on one entry. Pass exactly one of --text, --file or --clear."""
    if sum((text is not None, source is not None, clear)) != 1:
        raise click.ClickException("pass exactly one of --text, --file or --clear")
    experiment = _open_experiment(experiment_dir)
    if clear:
        new_text = ""
    elif text is not None:
        new_text = text
    else:
        # `open_file` reads `-` as standard input, which a plain read_text cannot.
        with click.open_file(str(source), encoding="utf-8-sig") as handle:
            new_text = handle.read()
    try:
        experiment.set_notes(kind, entry_id, new_text)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    where = f"{kind} {entry_id}" if entry_id else kind
    click.echo(f"notes {'cleared' if not new_text.strip() else 'saved'}: {where}")


# The ``keypoints`` group lives in its own module (§26 "Surfaces"): eight PRs add its
# subcommands, and each editing one region of this file is the merge conflict the build plan
# exists to avoid. Imported here, at the bottom, so that module never has to import this one.
from detail.cli_keypoints import dataset_keypoints as _dataset_keypoints  # noqa: E402
from detail.cli_keypoints import keypoints as _keypoints_group  # noqa: E402
from detail.cli_skeletons import skeleton as _skeleton_group  # noqa: E402

main.add_command(_keypoints_group)
# `dataset keypoints` (§12 *Distill keypoints*, the label stage on its own) shares its options
# with `keypoints distill`, so it is declared beside that command and registered here.
dataset.add_command(_dataset_keypoints)
# `process keypoint_detect` is the same command as `keypoints detect` (§26 *Surfaces* names
# both routes); one definition, registered twice, so the two cannot drift.
process.add_command(_keypoints_group.commands["detect"], "keypoint_detect")
process.add_command(_keypoints_group.commands["track"], "keypoint_track")
main.add_command(_skeleton_group)

# Hide the commands this build does not surface (§25), after every group is registered. Click's
# ``hidden`` keeps a command out of ``--help`` and out of the generated reference while leaving it
# executable, so a research sweep still runs against a build a user never sees it in.
surfaces.apply_cli_visibility(main)


if __name__ == "__main__":
    main()
