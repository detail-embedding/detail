"""The ``detail keypoints`` command group (§26 "Surfaces").

The one command group ``cli.py`` does not hold. Eight PRs add keypoint commands (§26
*Sequencing*), and each adding its subcommands to one region of an 8,800-line file is the
merge conflict the build plan exists to avoid — so the group lives here and ``cli.py`` only
registers it (``main.add_command``). Subcommands land in the fixed order below, one PR each;
a PR fills its own placeholder and touches no other.

This module imports nothing from ``detail.cli`` at the top: ``cli.py`` imports it, and the
reverse edge would be a cycle. Helpers a subcommand needs from ``cli.py`` (opening the
experiment, progress bars) are imported lazily inside the command body.
"""

from pathlib import Path

import click

from detail.api.keypoint_distill import DEFAULT_POSE_BASE
from detail.configs.keypoint_dataset import KeypointDatasetConfig
from detail.configs.keypoint_detect import KeypointDetectConfig
from detail.configs.keypoint_track import METHOD_CONFIGS, KeypointTrackConfig
from detail.configs.training import UltralyticsYOLOConfig

_EXPERIMENT_OPT = click.option(
    "--experiment",
    "experiment_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Experiment directory.",
)
_RECOVERABLE = (ValueError, FileNotFoundError, KeyError, IndexError)


def _open(experiment_dir: Path):
    from detail.cli import _open_experiment

    return _open_experiment(experiment_dir)


def _fail(exc: BaseException) -> click.ClickException:
    """A clean message: ``str(KeyError("x"))`` is ``"'x'"``, so take its argument."""
    if isinstance(exc, KeyError) and exc.args:
        return click.ClickException(str(exc.args[0]))
    return click.ClickException(str(exc))


@click.group()
def keypoints() -> None:
    """Keypoints (§26): example sets, the detect and track stages, export
    and distill. The detect and track stages land with their modules."""


# -- keypoints (§26): PR 2 -- `keypoints examples list|create|edit|duplicate|sample|add-frame|show|import|export|delete`
@keypoints.group("examples")
def examples() -> None:
    """Example sets: a skeleton plus a few frames with its points marked on them, the input
    a patch-matching keypoint method learns from.

    One set serves every recording. Create one on a library skeleton (`detail skeleton
    list`), another set's skeleton or a skeleton file; sample frames here, then mark them in
    the GUI's labeling screen; or prepare the YAML by hand and import it.
    """


@examples.command("list")
@_EXPERIMENT_OPT
@click.option(
    "--purpose",
    type=click.Choice(["example", "all"]),
    default="example",
    show_default=True,
    help="Which sets: the ones a method learns from, or every set in the experiment "
    "(which includes any audit set an earlier version made).",
)
def examples_list(experiment_dir: Path, purpose: str) -> None:
    """List the experiment's example sets."""
    proj = _open(experiment_dir)
    sets = proj.example_sets(None if purpose == "all" else purpose)
    if not sets:
        click.echo("no example sets")
        return
    for es in sets:
        counts = es.counts()
        click.echo(
            f"{es.id}  {es.name}  {es.purpose}  {counts['drawn']} frames, "
            f"{counts['labeled']} marked  skeleton {len(es.skeleton)} points"
        )


@examples.command("create")
@_EXPERIMENT_OPT
@click.argument("name", type=str)
@click.option(
    "--skeleton",
    "skeleton_ref",
    required=True,
    type=str,
    help="A library or preset skeleton id (see `detail skeleton list`), another example "
    "set's id (its skeleton is copied), or a skeleton YAML file.",
)
def examples_create(experiment_dir: Path, name: str, skeleton_ref: str) -> None:
    """Create an empty example set holding a copy of a skeleton."""
    proj = _open(experiment_dir)
    try:
        created = proj.create_example_set(name, skeleton_ref)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"created example set: {created.id}")


@examples.command("sample")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
@click.option(
    "--recording",
    "recordings",
    multiple=True,
    required=True,
    help="Recording name (or id) to sample from; repeatable.",
)
@click.option(
    "--tracking-run",
    "tracking_run_id",
    default=None,
    type=str,
    help="The tracking run to read in each recording; default: its only complete one.",
)
@click.option(
    "-n", "--frames", "n_frames", default=5, show_default=True, type=int, help="Frames."
)
@click.option(
    "--strategy",
    type=click.Choice(["varied", "spread"]),
    default="varied",
    show_default=True,
    help="varied: postures that differ, typical ones first. spread: one per time slice.",
)
@click.option(
    "--typical",
    "n_typical",
    default=5,
    show_default=True,
    type=int,
    help="varied only: typical frames an empty set opens with.",
)
@click.option(
    "--from-recordings",
    default=20,
    show_default=True,
    type=int,
    help="varied only: read at most this many recordings, spread through the set; 0 = all.",
)
@click.option(
    "--track", "track_ids", multiple=True, type=int, help="Only these track ids."
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--device",
    default=None,
    type=str,
    help="varied only: device for the DINOv3 pass (cpu, cuda, mps); default: the best present.",
)
def examples_sample(
    experiment_dir: Path,
    set_id: str,
    recordings: tuple[str, ...],
    tracking_run_id: str | None,
    n_frames: int,
    strategy: str,
    n_typical: int,
    from_recordings: int,
    track_ids: tuple[int, ...],
    seed: int,
    device: str | None,
) -> None:
    """Copy tracked frames from the chosen recordings into SET_ID, unmarked."""
    from detail.cli import _StageProgress

    proj = _open(experiment_dir)
    bar = _StageProgress()
    try:
        updated = proj.sample_example_frames(
            set_id,
            {ref: tracking_run_id for ref in recordings},
            n_frames,
            strategy=strategy,
            n_typical=n_typical,
            from_recordings=from_recordings or None,
            track_ids=list(track_ids) or None,
            seed=seed,
            device=device,
            progress=bar,
        )
    except _RECOVERABLE as e:
        raise _fail(e)
    finally:
        bar.close()
    click.echo(f"{set_id}: {len(updated.frames)} frames")


@examples.command("add-frame")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
@click.option("--recording", required=True, type=str, help="Recording name (or id).")
@click.option("--tracking-run", "tracking_run_id", required=True, type=str)
@click.option(
    "--part", "part_id", required=True, type=str, help="Video part id, e.g. part_001."
)
@click.option(
    "--frame", "frame_idx", required=True, type=int, help="Frame within the part."
)
@click.option("--track", "track_id", required=True, type=int, help="Track id.")
def examples_add_frame(
    experiment_dir: Path,
    set_id: str,
    recording: str,
    tracking_run_id: str,
    part_id: str,
    frame_idx: int,
    track_id: int,
) -> None:
    """Copy one chosen frame into SET_ID (the track must have a subject on it)."""
    proj = _open(experiment_dir)
    try:
        updated = proj.add_example_frame(
            set_id, recording, tracking_run_id, part_id, frame_idx, track_id
        )
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"{set_id}: {len(updated.frames)} frames")


@examples.command("show")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
def examples_show(experiment_dir: Path, set_id: str) -> None:
    """Show a set's skeleton, frames and marks."""
    proj = _open(experiment_dir)
    try:
        es = proj.example_set(set_id)
    except _RECOVERABLE as e:
        raise _fail(e)
    names = proj.recording_names()
    click.echo(f"{es.id}  {es.name}  purpose={es.purpose}  created {es.created_at}")
    click.echo(f"skeleton ({es.skeleton.digest()}): {', '.join(es.skeleton.names)}")
    if es.skeleton.edges:
        click.echo("  edges: " + ", ".join(f"{a}-{b}" for a, b in es.skeleton.edges))
    if es.skeleton.pairs:
        click.echo("  pairs: " + ", ".join(f"{a}/{b}" for a, b in es.skeleton.pairs))
    for i, frame in enumerate(es.frames):
        visible = sum(1 for m in frame.marks if m.visible)
        hidden = len(frame.marks) - visible
        state = "marked" if es.labeled(i) else "pending"
        how = frame.sampled_by or "picked"
        click.echo(
            f"[{i}] {names.get(frame.recording_id, frame.recording_id)} {frame.part_id} "
            f"frame {frame.frame_idx} track {frame.track_id}  {state}: {visible} placed, "
            f"{hidden} not visible  ({how})"
        )
    counts = es.counts()
    click.echo(
        f"{counts['drawn']} frames: {counts['labeled']} marked, {counts['pending']} pending"
    )


@examples.command("import")
@_EXPERIMENT_OPT
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def examples_import(experiment_dir: Path, path: Path) -> None:
    """Import a prepared set (an examples.yaml with its frame images beside it) or a bare
    skeleton file (names, edges, pairs), which becomes an empty set."""
    proj = _open(experiment_dir)
    try:
        created = proj.import_example_set(path)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"imported example set: {created.id} ({len(created.frames)} frames)")


@examples.command("edit")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
@click.option("--name", default=None, type=str, help="The set's new name.")
@click.option(
    "--skeleton",
    "skeleton_ref",
    default=None,
    type=str,
    help="Change the set's skeleton (a library or preset id, or a skeleton YAML file); "
    "refused once any frame has a mark.",
)
def examples_edit(
    experiment_dir: Path, set_id: str, name: str | None, skeleton_ref: str | None
) -> None:
    """Rename SET_ID, or change its skeleton until its first mark (after that, make a new
    set for another skeleton)."""
    proj = _open(experiment_dir)
    try:
        proj.update_example_set(set_id, name=name, skeleton=skeleton_ref)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"updated example set: {set_id}")


@examples.command("duplicate")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
@click.option("--name", default=None, type=str, help="The copy's name.")
def examples_duplicate(experiment_dir: Path, set_id: str, name: str | None) -> None:
    """A new, empty set with a copy of SET_ID's skeleton (how a used set's points change)."""
    proj = _open(experiment_dir)
    try:
        copy = proj.duplicate_example_set(set_id, name=name)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"created example set: {copy.id}")


@examples.command("export")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="An empty folder to write examples.yaml and its frame images into.",
)
def examples_export(experiment_dir: Path, set_id: str, out_dir: Path) -> None:
    """Write SET_ID to a folder another experiment can `examples import`."""
    proj = _open(experiment_dir)
    try:
        written = proj.export_example_set(set_id, out_dir)
    except (*_RECOVERABLE, FileExistsError) as e:
        raise _fail(e)
    click.echo(f"wrote {written}")


@examples.command("delete")
@_EXPERIMENT_OPT
@click.argument("set_id", type=str)
def examples_delete(experiment_dir: Path, set_id: str) -> None:
    """Delete a set no keypoint run lists."""
    proj = _open(experiment_dir)
    try:
        proj.delete_example_set(set_id)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"deleted example set: {set_id}")


# -- keypoints (§26): PR 4 -- `keypoints detect` (also `process keypoint_detect`, cli.py)
_KD = KeypointDetectConfig.model_fields


@keypoints.command("detect")
@_EXPERIMENT_OPT
@click.option(
    "--recording",
    "recording_ids",
    required=True,
    multiple=True,
    type=str,
    help="Recording name (or id) to include (repeatable; at least one).",
)
@click.option(
    "--detection",
    "detection_run_id",
    default=None,
    type=str,
    help="Detection run id, complete in every selected recording. "
    + _KD["detection_run_id"].description,
)
@click.option(
    "--detection-ref",
    "detection_ref",
    default=None,
    help="RECORDING:det_NN — derive the detection kind from this run and resolve each "
    "recording's own match (alternative to --detection, for recordings with divergent ids).",
)
@click.option(
    "--model", "model_id", required=True, type=str, help=_KD["model_id"].description
)
@click.option(
    "--variant",
    "model_variant",
    default=None,
    type=str,
    help=_KD["model_variant"].description,
)
@click.option(
    "--example-set",
    "example_set_id",
    default=None,
    type=str,
    help=_KD["example_set_id"].description,
)
@click.option(
    "--min-pose-confidence",
    "min_pose_confidence",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help=_KD["min_pose_confidence"].description,
)
@click.option(
    "--min-detection-confidence",
    "min_detection_confidence",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help=_KD["min_detection_confidence"].description,
)
@click.option(
    "--decode",
    type=click.Choice(["skeleton", "per_point"]),
    default=None,
    help=_KD["decode"].description,
)
@click.option(
    "--side-rule",
    "side_rule",
    type=click.Choice(["auto", "midline", "hard", "soft", "off"]),
    default=None,
    help=_KD["side_rule"].description,
)
@click.option(
    "--crop-px", "crop_px", type=int, default=None, help=_KD["crop_px"].description
)
@click.option(
    "--every-n-frames",
    "every_n_frames",
    type=int,
    default=None,
    help=_KD["every_n_frames"].description,
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=_KD["batch_size"].default,
    show_default=True,
    help=_KD["batch_size"].description,
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    show_default=True,
    help="Compute backend.",
)
@click.option(
    "--device",
    "devices",
    multiple=True,
    type=int,
    help="Device index (repeatable for multi-GPU; cuda only).",
)
def keypoints_detect(
    experiment_dir: Path,
    recording_ids: tuple[str, ...],
    detection_run_id: str | None,
    detection_ref: str | None,
    model_id: str,
    model_variant: str | None,
    example_set_id: str | None,
    min_pose_confidence: float | None,
    min_detection_confidence: float | None,
    decode: str | None,
    side_rule: str | None,
    crop_px: int | None,
    every_n_frames: int | None,
    batch_size: int,
    backend: str,
    devices: tuple[int, ...],
) -> None:
    """Place keypoints on the animals a detection run found in the selected recordings
    (keypoint detect, §26); writes an experiment run (run_NNN). Every pose is stored per frame
    with no subject; ``keypoints track`` gives each one to a tracked animal. Point at the
    detections with either ``--detection det_NN`` (that id must be complete in every selected
    recording) or ``--detection-ref RECORDING:det_NN``. A method that learns from examples
    (dinov3_match) needs ``--example-set``; a setting left out is resolved against the set's
    skeleton and recorded on the run. The same command is ``process keypoint_detect``.
    """
    from detail.cli import (
        _check_status,
        _compute_target,
        _resolve_recording_refs,
        _spec_from_ref,
        _stream_processing_progress,
        _with_stage_progress,
    )
    from detail.core.protocols import ModelNotReadyError

    if bool(detection_run_id) == bool(detection_ref):
        raise click.ClickException("pass exactly one of --detection or --detection-ref")
    proj = _open(experiment_dir)
    config = {
        "model_id": model_id,
        "model_variant": model_variant,
        "example_set_id": example_set_id,
        "min_pose_confidence": min_pose_confidence,
        "min_detection_confidence": min_detection_confidence,
        "decode": decode,
        "side_rule": side_rule,
        "crop_px": crop_px,
        "every_n_frames": every_n_frames,
        "batch_size": batch_size,
        "compute": _compute_target(backend, devices),
    }
    if detection_ref:
        config["detection_spec"] = _spec_from_ref(proj, "detection", detection_ref)
    else:
        config["detection_run_id"] = detection_run_id
    try:
        refs = _resolve_recording_refs(proj, recording_ids)
        run = _with_stage_progress(
            lambda bar: proj.run_processing(
                "keypoint_detect", refs, config, progress=bar
            )
        )
    except (*_RECOVERABLE, ModelNotReadyError) as e:
        raise _fail(e)
    click.echo(f"submitted processing run: {run.id}")
    _stream_processing_progress(run, "keypoint_detect")
    _check_status(run, "processing")


# -- keypoints (§26): PR 4 -- `keypoints test`
# -- keypoints (§26): PR 5 -- `keypoints track` (also `process keypoint_track`, cli.py)
_KT = KeypointTrackConfig.model_fields


def _method_params(method_id: str, pairs: tuple[str, ...]) -> dict:
    """``NAME=VALUE`` pairs typed as the method's config model declares them (§6.5): the model
    is the single source for the type, so ``despike_window_s=0.2`` becomes a float and
    ``despike=false`` a bool without the command knowing either field."""
    model = METHOD_CONFIGS.get(method_id)
    if model is None:
        raise click.ClickException(
            f"unknown track method {method_id!r}; available: {sorted(METHOD_CONFIGS)}"
        )
    raw: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.ClickException(
                f"bad --method-param {pair!r}; expected NAME=VALUE"
            )
        name, value = pair.split("=", 1)
        if name not in model.model_fields:
            raise click.ClickException(
                f"{method_id} has no parameter {name!r}; it has "
                f"{sorted(model.model_fields)}"
            )
        raw[name] = value
    try:  # the model coerces strings ("0.2", "false", "none") in lax mode
        return model.model_validate(
            {k: (None if v.lower() in ("none", "null") else v) for k, v in raw.items()}
        ).model_dump(include=set(raw))
    except ValueError as exc:
        raise click.ClickException(str(exc))


@keypoints.command("track")
@_EXPERIMENT_OPT
@click.option(
    "--recording",
    "recording_ids",
    required=True,
    multiple=True,
    type=str,
    help="Recording name (or id) to include (repeatable; at least one).",
)
@click.option(
    "--detect",
    "detect_run_id",
    default=None,
    type=str,
    help="Keypoint detect run id, complete in every selected recording.",
)
@click.option(
    "--detect-ref",
    "detect_ref",
    default=None,
    help="RECORDING:kd_NN — derive the detect kind from this run and resolve each "
    "recording's own match (alternative to --detect, for recordings with divergent ids).",
)
@click.option(
    "--tracking",
    "tracking_run_id",
    default=None,
    type=str,
    help=_KT["tracking_run_id"].description,
)
@click.option(
    "--tracking-ref",
    "tracking_ref",
    default=None,
    help="RECORDING:track_NN — derive the tracking kind from this run and resolve each "
    "recording's own match (alternative to --tracking, for recordings with divergent ids).",
)
@click.option(
    "--match-min-inside",
    "match_min_inside",
    type=click.FloatRange(0.0, 1.0, min_open=True),
    default=_KT["match_min_inside"].default,
    show_default=True,
    help=_KT["match_min_inside"].description,
)
@click.option(
    "--match-mask-grow",
    "match_mask_grow",
    type=click.FloatRange(min=0.0),
    default=_KT["match_mask_grow"].default,
    show_default=True,
    help=_KT["match_mask_grow"].description,
)
@click.option(
    "--method",
    "method",
    type=click.Choice(sorted(METHOD_CONFIGS)),
    default=None,
    help=_KT["method"].description,
)
@click.option(
    "--method-param",
    "method_params",
    multiple=True,
    metavar="NAME=VALUE",
    help="A setting of the chosen --method, e.g. --method-param smooth_window_s=0.2 or "
    "--method-param despike=false; names and types come from the method's parameter page "
    "(`detail docs`). Needs --method.",
)
def keypoints_track(
    experiment_dir: Path,
    recording_ids: tuple[str, ...],
    detect_run_id: str | None,
    detect_ref: str | None,
    tracking_run_id: str | None,
    tracking_ref: str | None,
    match_min_inside: float,
    match_mask_grow: float,
    method: str | None,
    method_params: tuple[str, ...],
) -> None:
    """Give a keypoint detect run's poses to a tracking run's subjects by mask, then settle
    each subject's points over time (keypoint track, §26); writes an experiment run
    (run_NNN). Point at the detect runs with either ``--detect kd_NN`` (that id must be
    complete in every selected recording) or ``--detect-ref RECORDING:kd_NN``. Name the
    tracking run with ``--tracking track_NN`` or ``--tracking-ref RECORDING:track_NN``;
    without either, each recording uses the one complete tracking run on the detect run's
    detection run, and several are refused, named. The method defaults to the detect run's
    method's own (dinov3_match: temporal_decode); every setting left out takes the method's
    default and is recorded on the run. CPU only. The same command is ``process
    keypoint_track``."""
    from detail.cli import (
        _check_status,
        _resolve_recording_refs,
        _spec_from_ref,
        _stream_processing_progress,
        _with_stage_progress,
    )

    if bool(detect_run_id) == bool(detect_ref):
        raise click.ClickException("pass exactly one of --detect or --detect-ref")
    if tracking_run_id and tracking_ref:
        raise click.ClickException("pass one of --tracking or --tracking-ref, not both")
    if method_params and method is None:
        raise click.ClickException("--method-param needs --method")
    proj = _open(experiment_dir)
    config: dict = {
        "tracking_run_id": tracking_run_id,
        "match_min_inside": match_min_inside,
        "match_mask_grow": match_mask_grow,
        "method": method,
        "config": _method_params(method, method_params) if method else {},
    }
    if detect_ref:
        config["keypoint_detect_spec"] = _spec_from_ref(
            proj, "keypoint_detect", detect_ref
        )
    else:
        config["detect_run_id"] = detect_run_id
    if tracking_ref:
        config["tracking_spec"] = _spec_from_ref(proj, "tracking", tracking_ref)
    try:
        refs = _resolve_recording_refs(proj, recording_ids)
        run = _with_stage_progress(
            lambda bar: proj.run_processing(
                "keypoint_track", refs, config, progress=bar
            )
        )
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"submitted processing run: {run.id}")
    _stream_processing_progress(run, "keypoint_track")
    _check_status(run, "processing")


# -- keypoints (§26): PR 7 -- `keypoints export`
# -- keypoints (§26): PR 8 -- `keypoints test` and `keypoints distill` (and `dataset keypoints`, registered by cli.py)
_KDS = KeypointDatasetConfig.model_fields
_TRAIN = UltralyticsYOLOConfig.model_fields

#: The label stage's options, one declaration per ``KeypointDatasetConfig`` field, shared by
#: ``keypoints distill`` and ``dataset keypoints`` so the two cannot drift.
#: Which teacher, and what it needs: shared by `distill`, `dataset keypoints` and `test`.
_TEACHER_OPTIONS = (
    click.option(
        "--teacher",
        "teacher",
        required=True,
        type=click.Choice(["patch", "detect"]),
        help="What labels the frames, both unsmoothed: `patch`, the patch method itself run "
        "on the sampled frames (with --example-set and the tracking runs), or `detect`, "
        "finished keypoint detect runs' stored poses, each given to a tracked animal.",
    ),
    click.option(
        "--recording",
        "recording_ids",
        multiple=True,
        type=str,
        help="Recording name (or id) to label from (repeatable): its only complete "
        "tracking run for --teacher patch, its only complete keypoint detect run for "
        "--teacher detect.",
    ),
    click.option(
        "--detect-run",
        "detect_run_refs",
        multiple=True,
        metavar="RECORDING:kd_NN",
        help="--teacher detect: a named keypoint detect run, for a recording with several "
        "(repeatable; alternative to --recording).",
    ),
    click.option(
        "--tracking-run",
        "tracking_run_refs",
        multiple=True,
        metavar="RECORDING:track_NN",
        help="A named tracking run, for a recording with several (repeatable). --teacher "
        "patch: the run whose animals the method labels (alternative to --recording). "
        "--teacher detect: the run the detect run's poses are given to; left out, the one "
        "complete tracking run on the detect run's detection run.",
    ),
    click.option(
        "--example-set",
        "example_set",
        default=None,
        type=str,
        help="--teacher patch: the example set (ex_NN) the patch method learns from.",
    ),
    click.option(
        "--model",
        "model_id",
        default="dinov3_match",
        show_default=True,
        type=str,
        help="--teacher patch: the patch method.",
    ),
    click.option(
        "--model-variant",
        "model_variant",
        default=None,
        type=str,
        help="--teacher patch: the method's size (default: its own default).",
    ),
    click.option(
        "--label-backend",
        "label_backend",
        type=click.Choice(["cpu", "cuda", "mps"]),
        default=None,
        help="--teacher patch: where the method runs (default: CUDA if available).",
    ),
    click.option(
        "--label-device",
        "label_devices",
        multiple=True,
        type=int,
        help="--teacher patch: CUDA device index for labelling (repeatable).",
    ),
    # The patch method's settings: test and distill take the same ones, so a cut-off judged in
    # `keypoints test` is the one the distill labels with.
    click.option(
        "--decode",
        type=click.Choice(["skeleton", "per_point"]),
        default=None,
        help="--teacher patch: " + _KD["decode"].description,
    ),
    click.option(
        "--side-rule",
        "side_rule",
        type=click.Choice(["auto", "midline", "hard", "soft", "off"]),
        default=None,
        help="--teacher patch: " + _KD["side_rule"].description,
    ),
    click.option(
        "--crop-px",
        "crop_px",
        type=int,
        default=None,
        help="--teacher patch: " + _KD["crop_px"].description,
    ),
)

#: How frames are drawn: shared by the dataset commands and `test`, so a test draws as the
#: distill will.
_DRAW_OPTIONS = (
    click.option(
        "--exclude-audit-s",
        "exclude_audit_s",
        default=_KDS["exclude_audit_s"].default,
        show_default=True,
        type=float,
        help=_KDS["exclude_audit_s"].description,
    ),
    click.option(
        "--every-n-frames",
        "every_n_frames",
        default=_KDS["every_n_frames"].default,
        show_default=True,
        type=click.IntRange(min=1),
        help=_KDS["every_n_frames"].description,
    ),
)

_DATASET_OPTIONS = (
    click.option("--name", required=True, type=str, help="Dataset (and student) name."),
    *_TEACHER_OPTIONS,
    click.option(
        "--n-frames",
        "n_frames",
        required=True,
        type=click.IntRange(min=1),
        help=_KDS["n_frames"].description,
    ),
    click.option(
        "--val-split",
        "val_split",
        default=_KDS["val_split"].default,
        show_default=True,
        type=float,
        help=_KDS["val_split"].description,
    ),
    click.option(
        "--score-cutoff",
        "score_cutoff",
        default=None,
        type=float,
        help=_KDS["score_cutoff"].description,
    ),
    *_DRAW_OPTIONS,
)


def _dataset_options(command):
    for option in reversed(_DATASET_OPTIONS):
        command = option(command)
    return command


def _test_options(command):
    for option in reversed((*_TEACHER_OPTIONS, *_DRAW_OPTIONS)):
        command = option(command)
    return command


def _named_runs(recording_ids, refs, option: str, shape: str, what: str) -> dict:
    """``{recording ref: run id or None}`` from ``--recording`` and a ``REC:RUN`` option."""
    runs: dict = {ref: None for ref in recording_ids}
    for spec in refs:
        ref, sep, run_id = spec.partition(":")
        if not sep or not ref.strip() or not run_id.strip():
            raise click.ClickException(f"{option} must be {shape}, got {spec!r}")
        runs[ref.strip()] = run_id.strip()
    if not runs:
        raise click.ClickException(
            f"name at least one {what}: --recording <name> or {option} {shape}"
        )
    return runs


def _teacher_args(
    teacher: str,
    recording_ids,
    detect_run_refs,
    tracking_run_refs,
    example_set,
    model_id,
    model_variant,
    label_backend,
    label_devices,
    settings: dict,
) -> dict:
    """The API's teacher keywords from the CLI's options, refusing a mix."""
    settings = {name: value for name, value in settings.items() if value is not None}
    if teacher == "detect":
        if example_set or label_backend:
            raise click.ClickException(
                "--example-set and --label-backend are for --teacher patch; --teacher "
                "detect reads finished keypoint detect runs"
            )
        if settings:
            raise click.ClickException(
                "--decode, --side-rule and --crop-px are for --teacher patch; a detect "
                "run's settings are the ones it recorded"
            )
        detect_runs = _named_runs(
            recording_ids,
            detect_run_refs,
            "--detect-run",
            "RECORDING:kd_NN",
            "teacher detect run",
        )
        if not tracking_run_refs:
            return {"detect_runs": detect_runs}
        # Only the named ones: the API resolves each other recording's one tracking run.
        tracking = _named_runs(
            (), tracking_run_refs, "--tracking-run", "RECORDING:track_NN", ""
        )
        return {"detect_runs": detect_runs, "tracking": tracking}
    if detect_run_refs:
        raise click.ClickException(
            "--detect-run is for --teacher detect; --teacher patch takes --tracking-run"
        )
    if not example_set:
        raise click.ClickException(
            "--teacher patch learns from an example set: give --example-set ex_NN"
        )
    from detail.cli import _compute_target

    return {
        "example_set": example_set,
        "tracking": _named_runs(
            recording_ids,
            tracking_run_refs,
            "--tracking-run",
            "RECORDING:track_NN",
            "recording to label",
        ),
        "model_id": model_id,
        "model_variant": model_variant,
        "label_compute": (
            _compute_target(label_backend, label_devices) if label_backend else None
        ),
        **({"settings": settings} if settings else {}),
    }


@keypoints.command("test")
@_EXPERIMENT_OPT
@_test_options
@click.option(
    "--n",
    "n_frames",
    default=12,
    show_default=True,
    type=click.IntRange(min=1),
    help="Frames to test on, drawn the way the distill would draw them.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Another sample of the same kind.",
)
@click.option(
    "--cutoff",
    "cutoff",
    default=None,
    type=float,
    help="Report at this score cut-off (default: the method's own, or none).",
)
def keypoints_test(
    experiment_dir: Path,
    teacher: str,
    recording_ids: tuple[str, ...],
    detect_run_refs: tuple[str, ...],
    tracking_run_refs: tuple[str, ...],
    example_set: str | None,
    model_id: str,
    model_variant: str | None,
    label_backend: str | None,
    label_devices: tuple[int, ...],
    decode: str | None,
    side_rule: str | None,
    crop_px: int | None,
    exclude_audit_s: float,
    every_n_frames: int,
    n_frames: int,
    seed: int,
    cutoff: float | None,
) -> None:
    """Test a teacher before a distill: place points on about a dozen frames drawn the way
    the distill would, and check the method against the example set's own clicks, each example
    frame scored by the others (errors in pixels and % of body size). Writes nothing; the GUI's
    Test cut-off… shows the same."""
    from detail.cli import _StageProgress
    from detail.pipeline.keypoint_preview import cutoff_report

    proj = _open(experiment_dir)
    teacher_args = _teacher_args(
        teacher,
        recording_ids,
        detect_run_refs,
        tracking_run_refs,
        example_set,
        model_id,
        model_variant,
        label_backend,
        label_devices,
        {"decode": decode, "side_rule": side_rule, "crop_px": crop_px},
    )
    bar = _StageProgress()
    try:
        result = proj.test_keypoints(
            n_frames=n_frames,
            seed=seed,
            exclude_audit_s=exclude_audit_s,
            every_n_frames=every_n_frames,
            progress=bar,
            **teacher_args,
        )
    except _RECOVERABLE as e:
        raise _fail(e)
    finally:
        bar.close()
    if cutoff is None:
        cutoff = result.default_cutoff
    scale = (
        f"{result.score_meaning}, {result.score_low:g} to {result.score_high:g}"
        if result.score_meaning
        else "the method's score"
    )
    click.echo(
        f"{result.method} on {len(result.frames)} of {result.requested} frame(s) "
        f"({result.device or 'stored placements'}); scores are {scale}"
    )
    shown, placed = result.taught_visible(cutoff)
    click.echo(
        f"cut-off {('none' if cutoff is None else f'{cutoff:g}')}: {shown} of {placed} "
        "placed points would be taught visible"
    )
    if result.held_out is None:
        click.echo("held-out check: not available (the teacher has no example set)")
        return
    report = cutoff_report(result.held_out, cutoff)

    def side(label: str, values: dict) -> str:
        if values["median_px"] is None:
            return f"{label}: n = {values['n']} (too few for a median)"
        return (
            f"{label}: median error {values['median_px']:.1f} px "
            f"({values['median_pct']:.1f}% of body size), "
            f"{values['near'] * 100:.0f}% within 5% · n = {values['n']}"
        )

    click.echo(
        "held out against the example clicks (each example scored by the others):"
    )
    click.echo("  " + side("at or above the cut-off", report["overall"]["above"]))
    click.echo("  " + side("below the cut-off", report["overall"]["below"]))
    for point, split in report["points"].items():
        click.echo(f"  {point}: " + side("above", split["above"]))


@keypoints.command("distill")
@_EXPERIMENT_OPT
@_dataset_options
@click.option(
    "--base-model",
    "base_model",
    default=DEFAULT_POSE_BASE,
    show_default=True,
    type=str,
    help="Pretrained YOLO-pose weights the trainer fine-tunes (a -pose checkpoint).",
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
    "--train-backend",
    "train_backend",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default=None,
    help="Training backend (default: CUDA if available, else CPU).",
)
@click.option(
    "--train-device",
    "train_devices",
    multiple=True,
    type=int,
    help="Training CUDA device index (repeatable).",
)
def keypoints_distill(
    experiment_dir: Path,
    name: str,
    teacher: str,
    recording_ids: tuple[str, ...],
    detect_run_refs: tuple[str, ...],
    tracking_run_refs: tuple[str, ...],
    example_set: str | None,
    model_id: str,
    model_variant: str | None,
    label_backend: str | None,
    label_devices: tuple[int, ...],
    decode: str | None,
    side_rule: str | None,
    crop_px: int | None,
    n_frames: int,
    val_split: float,
    score_cutoff: float | None,
    exclude_audit_s: float,
    every_n_frames: int,
    base_model: str,
    trainer_id: str,
    epochs: int | None,
    batch_size: int | None,
    imgsz: int | None,
    patience: int | None,
    train_backend: str | None,
    train_devices: tuple[int, ...],
) -> None:
    """Distill a teacher into a fast YOLO-pose student: label, train, then register it as a
    keypoint method, as one durable run (§12). The teacher is the patch method run on the
    sampled frames (`--teacher patch`) or finished keypoint detect runs (`--teacher detect`).
    Blocks until complete (Ctrl-C detaches; the run keeps going, `distill attach` rejoins).
    """
    from detail.cli import _check_status, _compute_target, _stream_distill_progress

    proj = _open(experiment_dir)
    teacher_args = _teacher_args(
        teacher,
        recording_ids,
        detect_run_refs,
        tracking_run_refs,
        example_set,
        model_id,
        model_variant,
        label_backend,
        label_devices,
        {"decode": decode, "side_rule": side_rule, "crop_px": crop_px},
    )
    hyperparams: dict = {}
    for key, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("imgsz", imgsz),
        ("patience", patience),
    ):
        if value is not None:
            hyperparams[key] = value
    try:
        run = proj.distill_keypoints(
            name,
            n_frames=n_frames,
            **teacher_args,
            val_split=val_split,
            score_cutoff=score_cutoff,
            exclude_audit_s=exclude_audit_s,
            every_n_frames=every_n_frames,
            base_model=base_model,
            trainer_id=trainer_id,
            hyperparams=hyperparams or None,
            train_compute=(
                _compute_target(train_backend, train_devices) if train_backend else None
            ),
        )
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"submitted distill: {run.id}")
    _stream_distill_progress(run, "distill")
    _check_status(run, "distill")


@click.command("keypoints")
@_EXPERIMENT_OPT
@_dataset_options
def dataset_keypoints(
    experiment_dir: Path,
    name: str,
    teacher: str,
    recording_ids: tuple[str, ...],
    detect_run_refs: tuple[str, ...],
    tracking_run_refs: tuple[str, ...],
    example_set: str | None,
    model_id: str,
    model_variant: str | None,
    label_backend: str | None,
    label_devices: tuple[int, ...],
    decode: str | None,
    side_rule: str | None,
    crop_px: int | None,
    n_frames: int,
    val_split: float,
    score_cutoff: float | None,
    exclude_audit_s: float,
    every_n_frames: int,
) -> None:
    """Build a YOLO-pose dataset (the label stage of a keypoint distill, on its own), taught
    by the patch method or by finished keypoint detect runs; blocks until complete.
    Registered as `dataset keypoints`."""
    from detail.cli import _check_status, _stream_progress

    proj = _open(experiment_dir)
    teacher_args = _teacher_args(
        teacher,
        recording_ids,
        detect_run_refs,
        tracking_run_refs,
        example_set,
        model_id,
        model_variant,
        label_backend,
        label_devices,
        {"decode": decode, "side_rule": side_rule, "crop_px": crop_px},
    )
    try:
        run = proj.keypoint_dataset(
            name,
            n_frames=n_frames,
            **teacher_args,
            val_split=val_split,
            score_cutoff=score_cutoff,
            exclude_audit_s=exclude_audit_s,
            every_n_frames=every_n_frames,
        )
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"submitted keypoint dataset: {run.id}")
    _stream_progress(run, "keypoint dataset")
    _check_status(run, "keypoint dataset")


@keypoints.command("export")
@_EXPERIMENT_OPT
@click.option(
    "--run",
    "run_id",
    required=True,
    help="The experiment run (run_NNN) of a keypoint track or keypoint detect module.",
)
@click.option(
    "--target",
    "target_ids",
    multiple=True,
    type=click.Choice(
        [
            "keypoints",
            "keypoints_dlc",
            "keypoints_sleap",
            "keypoints_nwb",
            "keypoints_video",
        ]
    ),
    default=("keypoints",),
    show_default=True,
    help="What to write (repeatable; several go into one folder). See `export list`.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where to put the export folder. The folder itself is named for you.",
)
@click.option(
    "--source",
    "source",
    type=click.Choice(["track", "detect"]),
    default=None,
    help="On a track run: the tracked points (the default) or the detect run's poses "
    "before tracking, for a before/after comparison. A detect run always exports its own, "
    "unassigned: no track, each pose numbered within its frame.",
)
@click.option(
    "--cutoff",
    "cutoff",
    type=float,
    default=None,
    help="The display cut-off: a point scoring under it is written not visible (its "
    "coordinates kept). Default: the detect method's own cut-off.",
)
@click.option(
    "--blank-below-cutoff/--keep-below-cutoff",
    "blank_below_cutoff",
    default=None,
    help="Write a point under the cut-off as missing (NaN) in every format, rather than "
    "keeping its coordinates and marking it not visible.",
)
@click.option(
    "--overlay/--points-only",
    "overlay",
    default=None,
    help="keypoints_video: also draw the tracking run's masks and boxes under the points.",
)
def keypoints_export(
    experiment_dir: Path,
    run_id: str,
    target_ids: tuple[str, ...],
    out_dir: Path,
    source: str | None,
    cutoff: float | None,
    blank_below_cutoff: bool | None,
    overlay: bool | None,
) -> None:
    """Write a keypoint run's points for other tools (§26): DETAIL's long CSV
    (``keypoints``), DeepLabCut CSV + H5 (``keypoints_dlc``), the SLEAP analysis HDF5
    (``keypoints_sleap``), NWB with ndx-pose (``keypoints_nwb``), or the footage with the
    points drawn on (``keypoints_video``). One file per video part, named by recording; DLC
    and SLEAP files carry no time, so ``timestamps.csv`` goes beside them; every export ships
    ``params.csv`` and ``coverage.csv``. The same targets are on ``export run`` and the GUI's
    Export control; this command adds the keypoint options."""
    from detail.api import exporters
    from detail.cli import _echo_export_result, _stream_progress

    proj = _open(experiment_dir)
    # Only what was asked for: a default sent as an option would override the target's own.
    options: dict = {}
    if source is not None:
        options["source"] = source
    if cutoff is not None:
        options["cutoff"] = cutoff
    if blank_below_cutoff is not None:
        options["blank_below_cutoff"] = blank_below_cutoff
    if overlay is not None:
        options["overlay"] = overlay
    try:
        handle = proj.run_export("run", run_id, list(target_ids), out_dir, **options)
    except (exporters.ExportError, *_RECOVERABLE) as e:
        raise _fail(e)
    click.echo(f"submitted export: {handle.id}")
    _stream_progress(handle, "export")
    _echo_export_result(proj, handle.id)


# -- keypoints (§26): PR 8 -- `keypoints distill`
