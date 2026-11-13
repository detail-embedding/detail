"""Distill keypoints (§12 *Distill keypoints*): the API behind ``Experiment.keypoint_dataset``
and ``Experiment.distill_keypoints``, delegated to the way ``api/keypoints.py`` is.

A keypoint distill is the keypoint twin of the SAM3 → YOLO distill: one durable run of
``kind: keypoints`` whose stages are **label → train → register**, on the same detached
coordinator, the same record store and the same run page. Only the stage bodies differ: the
label stage is ``pipeline/keypoint_dataset.py``, training is the §12 trainer unchanged on a
``-pose`` base, and register promotes the student as ``<name> <distill_NNN>`` with
``role: keypoints``, its skeleton copy, the weights' hash and its teacher, so the keypoint
method picker can offer it.

**One of two teachers, both raw** (§12; no smoothing over time, the user's decision
2026-10-09): finished ``keypoint_detect`` runs (``detect_runs=``), or the patch
method itself (``example_set=`` + ``tracking=``), run on the sampled frames. The patch teacher
is resolved and **copied at launch** (the example-set snapshot keypoint detect records, the
method's settings, the tracking runs), so the label stage teaches from what was checked even if
the set is edited afterwards.

Either teacher's instances are **unassigned** (§26, revised 2026-10-21) and are given to the
subjects of a tracking run by §26 *Assignment*'s function, so both teacher records carry
``tracking_runs`` (``{recording_id: tracking run id}``) and the assignment's ``match``
settings. A detect-run teacher's tracking run is resolved the way keypoint track resolves its
own: the one complete tracking run on the detect run's detection run, or the one ``tracking=``
names; several and none named is refused, naming them.

Every defaulted input is resolved here and written into the job (Tenet 9): the score cut-off
comes from the teacher's method when the caller leaves it unset, and a method with no default
(the patch method) refuses rather than guesses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from detail.configs.keypoint_dataset import KeypointDatasetConfig
from detail.configs.tracking import DEFAULT_TRACKER
from detail.core.keypoints import KeypointDatasetJob
from detail.core.registries import KEYPOINT_MODELS, TRAINERS
from detail.core.utils import (
    check_cuda_indices,
    default_compute,
    effective_backend,
    utc_timestamp,
)
from detail.storage import datasets as datasets_store
from detail.storage import distill_runs as distill_store
from detail.storage import example_sets
from detail.storage import experiment as experiment_store
from detail.storage import guidance
from detail.storage import keypoint_runs
from detail.storage import models as models_store
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import KeypointDetectRun

if TYPE_CHECKING:
    from detail.api.experiment import Experiment

#: The base checkpoint a keypoint distill trains from unless told otherwise (§12).
DEFAULT_POSE_BASE = "yolo11s-pose.pt"
KIND = "keypoints"
STAGES = ("label", "train", "register")


def _resolve_runs(
    experiment: "Experiment", refs, *, field: str, what: str, first: str
) -> dict[str, str]:
    """``{recording_id: run id}`` of one kind of complete run (``field`` in recording.yaml),
    from recording names or ids, each with a run id or ``None`` for the recording's only one.
    """
    pairs = (
        {ref: None for ref in refs}
        if isinstance(refs, (list, tuple))
        else dict(refs or {})
    )
    if not pairs:
        raise ValueError(f"a keypoint dataset needs at least one {what}")
    names = experiment.recording_names()
    out: dict[str, str] = {}
    for ref, run_id in pairs.items():
        recording = experiment.resolve_recording(ref)
        data = recording_store.read_recording(recording.dir)
        complete = [
            str(r["id"])
            for r in data.get(field, [])
            if str(r.get("status", "complete")) == "complete"
        ]
        name = names.get(recording.id, recording.id)
        if run_id:
            if str(run_id) not in complete:
                raise ValueError(
                    f"recording {name!r} has no complete {what} {run_id!r}; "
                    f"it has {complete or 'none'}"
                )
            out[recording.id] = str(run_id)
            continue
        if not complete:
            raise ValueError(f"recording {name!r} has no complete {what}; {first}")
        if len(complete) > 1:
            raise ValueError(
                f"recording {name!r} has {len(complete)} {what}s "
                f"({', '.join(complete)}); say which one to use"
            )
        out[recording.id] = complete[0]
    return out


def resolve_detect_runs(experiment: "Experiment", detect_runs) -> dict[str, str]:
    """``{recording_id: keypoint_detect run id}`` from recording names or ids, each with a run
    id or ``None`` for the recording's only complete one.

    Raises:
        KeyError: An unknown recording reference.
        ValueError: A recording with no complete detect run, several and none named, or a
            named run that is not complete.
    """
    return _resolve_runs(
        experiment,
        detect_runs,
        field="keypoint_detect_runs",
        what="keypoint detect run",
        first=guidance.missing_keypoints_hint("keypoint detect"),
    )


def resolve_tracking_runs(experiment: "Experiment", tracking) -> dict[str, str]:
    """``{recording_id: tracking run id}`` for the patch-method teacher: the tracking runs
    whose boxes (and masks) it is given, from recording names or ids, each with a run id or
    ``None`` for the recording's only complete one.

    Raises:
        KeyError: An unknown recording reference.
        ValueError: A recording with no complete tracking run, several and none named, or a
            named run that is not complete.
    """
    return _resolve_runs(
        experiment,
        tracking,
        field="tracking_runs",
        what="tracking run",
        first="run detection and tracking on it first",
    )


def _default_match() -> dict:
    """The assignment's settings a teacher records (§26 *Assignment*; policy, not
    measured). Resolved here so the record says what assigned its labels (Tenet 9)."""
    from detail.pipeline import pose_match

    return {
        "min_inside": float(pose_match.DEFAULT_MIN_INSIDE),
        "mask_grow": float(pose_match.DEFAULT_MASK_GROW),
    }


def _tracking_on(data: dict, detection_run_id: str) -> list[str]:
    """The complete tracking runs of a recording built on one detection run."""
    return [
        str(r["id"])
        for r in data.get("tracking_runs", [])
        if str(r.get("status", "complete")) == "complete"
        and str(r.get("detection_run_id")) == str(detection_run_id)
    ]


def resolve_detect_tracking(
    experiment: "Experiment", detect_runs: dict[str, str], tracking=None
) -> dict[str, str]:
    """``{recording_id: tracking run id}``: per detect run, the tracking run whose subjects
    its instances are given to — the one ``tracking`` names for the recording (recording
    names or ids → run id), else the one complete tracking run on the detect run's detection
    run (§26: never "the first", which would assign differently tomorrow).

    Raises:
        ValueError: A named tracking run that is not complete or not on the detect run's
            detection run; none on it; several and none named.
    """
    named: dict[str, str] = {}
    if isinstance(tracking, dict):
        for ref, run_id in tracking.items():
            if run_id:
                named[experiment.resolve_recording(ref).id] = str(run_id)
    names = experiment.recording_names()
    out: dict[str, str] = {}
    for recording_id, detect_run_id in detect_runs.items():
        recording_dir = experiment_store.recording_path(experiment.dir, recording_id)
        detection_run_id = KeypointDetectRun(
            recording_dir, detect_run_id
        ).detection_run_id
        data = recording_store.read_recording(recording_dir)
        on = _tracking_on(data, detection_run_id)
        name = names.get(recording_id, recording_id)
        where = (
            f"detection run {detection_run_id!r} (the one keypoint detect run "
            f"{detect_run_id!r} read)"
        )
        if recording_id in named:
            if named[recording_id] not in on:
                raise ValueError(
                    f"recording {name!r} has no complete tracking run "
                    f"{named[recording_id]!r} on {where}; it has {on or 'none'}"
                )
            out[recording_id] = named[recording_id]
            continue
        if not on:
            raise ValueError(
                f"recording {name!r} has no complete tracking run on {where}: a detect "
                "run's poses are given to a tracking run's subjects; run tracking first"
            )
        if len(on) > 1:
            from detail.pipeline.keypoint_track import describe_tracking_choice

            entries = [r for r in data.get("tracking_runs", []) if str(r["id"]) in on]
            raise ValueError(
                f"recording {name!r} has {len(on)} complete tracking runs on {where}: "
                f"{describe_tracking_choice(entries)}"
            )
        out[recording_id] = on[0]
    return out


def _resolve_detect_teacher(experiment: "Experiment", detect_runs, tracking) -> dict:
    """The detect-run teacher, checked: the runs (one skeleton across them), the tracking
    run per recording their instances are assigned to, and the assignment's settings."""
    resolved = resolve_detect_runs(experiment, detect_runs)
    digest = None
    for recording_id, run_id in resolved.items():
        recording_dir = experiment_store.recording_path(experiment.dir, recording_id)
        this = KeypointDetectRun(recording_dir, run_id).skeleton.digest()
        if digest is not None and this != digest:
            raise ValueError(
                f"keypoint detect run {run_id!r} of {recording_id!r} has a different "
                "skeleton from the other teachers; a dataset teaches one skeleton"
            )
        digest = this
    return {
        "kind": "keypoint_detect",
        "detect_runs": resolved,
        "tracking_runs": resolve_detect_tracking(experiment, resolved, tracking),
        "match": _default_match(),
    }


def _resolve_patch_teacher(
    experiment: "Experiment",
    example_set: str,
    tracking,
    model_id: str,
    model_variant: str | None,
    label_compute,
    settings: dict | None = None,
    class_example_sets: dict[str, str] | None = None,
):
    """The patch-method teacher, checked and copied the way keypoint detect's ``validate``
    checks and records it (``keypoint_detect.resolve_method``): the set runnable, the method
    installed and able to run on ``label_compute``, the example-set snapshot, the settings.
    ``settings`` are keypoint detect's method settings (``decode`` / ``side_rule`` /
    ``crop_px``); one left out is the method's own default, as on a detect run.
    ``class_example_sets`` (``{class: set}``, in place of ``example_set``) matches each
    detection against its own class's set, as a per-class detect run does (§26).
    """
    from detail.core.protocols import ModelNotReadyError
    from detail.pipeline.keypoint_detect import METHOD_SETTINGS, resolve_method

    cls = KEYPOINT_MODELS.get(model_id)
    if cls is None or not getattr(cls, "needs_examples", False):
        learners = sorted(
            m for m, c in KEYPOINT_MODELS.items() if getattr(c, "needs_examples", False)
        )
        raise ValueError(
            f"the patch-method teacher is a method that learns from an example set; "
            f"{model_id!r} is not one (available: {learners})"
        )
    unknown = sorted(set(settings or {}) - set(METHOD_SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown method setting(s) {unknown}; the settings are {list(METHOD_SETTINGS)}"
        )
    if not tracking:
        raise ValueError(
            "the patch-method teacher needs the tracking runs whose boxes it labels "
            "(tracking: recording names, or {recording: tracking run})"
        )
    tracking_runs = resolve_tracking_runs(experiment, tracking)
    compute = label_compute or default_compute(cls.supported_backends)
    method = {
        "model_id": model_id,
        "model_variant": model_variant,
        "example_set_id": example_set,
        "class_example_sets": class_example_sets,
        **{name: (settings or {}).get(name) for name in METHOD_SETTINGS},
    }
    try:
        resolve_method(experiment.dir, method, compute)
    except ModelNotReadyError as exc:
        # The model's own message already says where to install it.
        raise ValueError(f"{model_id!r} is not ready to run: {exc}") from exc
    teacher = {
        "kind": "model",
        "model_id": model_id,
        "model_variant": method["model_variant"],
        "settings": {name: method[name] for name in METHOD_SETTINGS},
        "example_set": method["example_set"],
        "example_sets": method.get("example_sets"),
        "class_name": method.get("class_name"),
        "skeleton": method["skeleton"],
        "skeleton_ref": method.get("skeleton_ref"),
        "tracking_runs": tracking_runs,
        "match": _default_match(),
    }
    return teacher, compute


def teacher_info(
    experiment: "Experiment", recording_id: str, detect_run_id: str
) -> dict:
    """What a form shows about a teacher detect run: its ``method``, its
    score's ``meaning`` / ``low`` / ``high`` and ``default_cutoff`` (``None`` when the method
    sets none), the ``example_set`` (``{id, name}``) the detect run learned from, or None,
    its ``detection_run_id`` and the ``tracking_run_ids`` its poses can be assigned to (the
    complete tracking runs on that detection run; the teacher needs exactly one, or one
    named). Reads three small records; off the UI thread in a loop over runs."""
    recording_dir = experiment_store.recording_path(experiment.dir, recording_id)
    run = KeypointDetectRun(recording_dir, detect_run_id)
    detect = run.meta
    model_id = detect["model_id"]
    from detail.models.resolve import keypoint_method_class

    cls = keypoint_method_class(experiment.dir, model_id)
    spec = None if cls is None else cls.score_spec
    used = keypoint_runs.detect_example_sets(detect)
    examples = used.get(None)
    return {
        "method": str(model_id),
        "default_cutoff": None if spec is None else spec.default_cutoff,
        "meaning": None if spec is None else spec.meaning,
        "low": None if spec is None else spec.low,
        "high": None if spec is None else spec.high,
        "example_set": (
            None
            if examples is None
            else {"id": examples.get("id"), "name": examples.get("name")}
        ),
        # a per-class run's sets, {class: {id, name}}; empty for one set or none
        "example_sets": {
            cls_: {"id": snap.get("id"), "name": snap.get("name")}
            for cls_, snap in used.items()
            if cls_ is not None
        },
        "detection_run_id": run.detection_run_id,
        "tracking_run_ids": _tracking_on(
            recording_store.read_recording(recording_dir), run.detection_run_id
        ),
    }


def teacher_cutoff(experiment: "Experiment", recording_id: str, detect_run_id: str):
    """The default score cut-off of a detect run's method — a registered method's
    ``score_spec.default_cutoff``, or a trained student's — ``None`` when it sets none.
    """
    return teacher_info(experiment, recording_id, detect_run_id)["default_cutoff"]


def resolve_teacher(
    experiment: "Experiment",
    detect_runs=None,
    *,
    example_set: str | None = None,
    tracking=None,
    model_id: str = "dinov3_match",
    model_variant: str | None = None,
    label_compute=None,
    settings: dict | None = None,
    class_example_sets: dict[str, str] | None = None,
):
    """``(teacher, compute)`` from the same keywords :func:`distill_keypoints` takes, checked:
    ``{kind: keypoint_detect, detect_runs, tracking_runs, match}`` (one skeleton across them;
    ``tracking`` optionally names each recording's tracking run; compute None) or the patch
    method's resolved copy and the compute it runs on. What :func:`keypoint_test` runs.

    Raises:
        ValueError: No teacher or two; a detect-run or patch-teacher refusal (see
            :func:`plan_keypoint_dataset`).
    """
    class_example_sets = experiment.resolve_class_refs(
        {"class_example_sets": class_example_sets}
    )["class_example_sets"]
    patch = example_set is not None or bool(class_example_sets)
    if (detect_runs is None) == (not patch) or (example_set and class_example_sets):
        raise ValueError(
            "give one teacher: detect_runs (finished keypoint detect runs), or example_set "
            "— or class_example_sets, one set per class — with tracking (the patch method "
            "run on the sampled frames)"
        )
    if patch:
        from detail.models.resolve import ensure_keypoint_methods

        ensure_keypoint_methods()  # `keypoint_test` reaches here with nothing imported
        return _resolve_patch_teacher(
            experiment,
            example_set,
            tracking,
            model_id,
            model_variant,
            label_compute,
            settings,
            class_example_sets,
        )
    if settings:
        raise ValueError(
            "method settings are for the patch-method teacher; a detect-run teacher's "
            "settings are the ones its run recorded"
        )
    return _resolve_detect_teacher(experiment, detect_runs, tracking), label_compute


def keypoint_test(
    experiment: "Experiment",
    detect_runs=None,
    *,
    example_set: str | None = None,
    tracking=None,
    model_id: str = "dinov3_match",
    model_variant: str | None = None,
    label_compute=None,
    settings: dict | None = None,
    n_frames: int = 12,
    seed: int = 0,
    every_n_frames: int = 1,
    exclude_audit_s: float = 1.0,
    progress=None,
    class_example_sets: dict[str, str] | None = None,
):
    """Test a teacher before a distill (*Test cut-off…* / *Test keypoints…*): about
    ``n_frames`` frames drawn the way the distill would, the teacher's placements on them, and
    the held-out check against the example clicks. Writes nothing; ``seed`` draws another
    sample. ``settings`` are the patch method's ``decode`` / ``side_rule`` / ``crop_px``, so
    the detect form's *Test keypoints…* places points the way its run will;
    ``class_example_sets`` (``{class: set}``) tests a per-class detect form the same way.
    Returns ``pipeline.keypoint_preview.TeacherTest``.

    Raises:
        ValueError: What :func:`resolve_teacher` refuses, or no frame to test on.
    """
    from detail.pipeline.keypoint_preview import run_teacher_test

    teacher, compute = resolve_teacher(
        experiment,
        detect_runs,
        example_set=example_set,
        tracking=tracking,
        model_id=model_id,
        model_variant=model_variant,
        label_compute=label_compute,
        settings=settings,
        class_example_sets=class_example_sets,
    )
    return run_teacher_test(
        experiment.dir,
        teacher,
        compute=compute,
        n_frames=n_frames,
        seed=seed,
        every_n_frames=every_n_frames,
        exclude_audit_s=exclude_audit_s,
        progress=progress,
    )


def plan_keypoint_dataset(
    experiment: "Experiment",
    name: str,
    detect_runs=None,
    n_frames: int | None = None,
    *,
    val_split: float = 0.15,
    score_cutoff: float | None = None,
    exclude_audit_s: float = 1.0,
    every_n_frames: int = 1,
    student_input: str = "whole_frame",
    student_crop_px: int = 448,
    example_set: str | None = None,
    tracking=None,
    model_id: str = "dinov3_match",
    model_variant: str | None = None,
    label_compute=None,
    settings: dict | None = None,
    teacher: dict | None = None,
    class_example_sets: dict[str, str] | None = None,
):
    """Everything a label stage needs, checked and resolved: ``(config, teacher, cutoff,
    compute)``. ``teacher`` is ``{kind: keypoint_detect, detect_runs, tracking_runs, match}``
    (one skeleton across the runs) or the patch method's resolved copy (``kind: model``);
    ``compute`` is where a patch teacher runs (None for detect runs, which need no compute).

    Exactly one teacher: ``detect_runs`` (with ``tracking`` optionally naming the tracking run
    each recording's poses are assigned to), **or** ``example_set`` — or
    ``class_example_sets``, one set per detection class (§26) — with ``tracking`` (the patch
    method, ``model_id``), **or** an already-resolved ``teacher`` (a distill record replaying
    what it checked at launch).

    Raises:
        ValueError: An empty name, a config outside its bounds, no teacher or two, teacher runs
            with different skeletons, an unrunnable example set or an uninstalled method, or
            no cut-off (the method sets no default and none was given).
    """
    if not str(name).strip():
        raise ValueError("a keypoint dataset needs a name")
    patch_sets = example_set is not None or bool(class_example_sets)
    given = [
        detect_runs is not None,
        patch_sets,
        teacher is not None and teacher.get("kind") == "model",
    ]
    if sum(given) != 1 or (example_set is not None and class_example_sets):
        raise ValueError(
            "give one teacher: detect_runs (finished keypoint detect runs), or example_set "
            "— or class_example_sets, one set per class — with tracking (the patch method "
            "run on the sampled frames)"
        )
    if settings and not patch_sets:
        raise ValueError(
            "method settings are for the patch-method teacher given by example_set; a "
            "detect-run teacher's settings are the ones its run recorded, and a resolved "
            "teacher carries its own"
        )
    cfg = KeypointDatasetConfig.model_validate(
        {
            "n_frames": n_frames,
            "val_split": val_split,
            "score_cutoff": score_cutoff,
            "exclude_audit_s": exclude_audit_s,
            "every_n_frames": every_n_frames,
            "student_input": student_input,
            "student_crop_px": student_crop_px,
        }
    )
    if detect_runs is None:
        if teacher is None:
            teacher, compute = _resolve_patch_teacher(
                experiment,
                example_set,
                tracking,
                model_id,
                model_variant,
                label_compute,
                settings,
                class_example_sets,
            )
        else:
            from detail.core.dataclasses import ComputeTarget

            compute = (
                label_compute
                if label_compute is not None
                else (
                    ComputeTarget.from_dict(teacher["compute"])
                    if teacher.get("compute")
                    else default_compute(
                        KEYPOINT_MODELS[teacher["model_id"]].supported_backends
                    )
                )
            )
        cutoff = cfg.score_cutoff
        if cutoff is None:
            cls = KEYPOINT_MODELS.get(teacher["model_id"])
            cutoff = None if cls is None else cls.score_spec.default_cutoff
        if cutoff is None:
            raise ValueError(
                f"{teacher['model_id']} sets no default score cut-off; pass score_cutoff "
                "(a point scored below it is taught as not visible)"
            )
        return cfg, teacher, float(cutoff), compute
    cfg = KeypointDatasetConfig.model_validate(
        {
            "n_frames": n_frames,
            "val_split": val_split,
            "score_cutoff": score_cutoff,
            "exclude_audit_s": exclude_audit_s,
            "every_n_frames": every_n_frames,
            "student_input": student_input,
            "student_crop_px": student_crop_px,
        }
    )
    teacher = _resolve_detect_teacher(experiment, detect_runs, tracking)
    cutoff = cfg.score_cutoff
    for recording_id, run_id in teacher["detect_runs"].items():
        if cutoff is None:
            cutoff = teacher_cutoff(experiment, recording_id, run_id)
    if cutoff is None:
        raise ValueError(
            "the teacher's method sets no default score cut-off; pass score_cutoff (a point "
            "below it is taught as not visible)"
        )
    return cfg, teacher, float(cutoff), None


def keypoint_dataset(
    experiment: "Experiment",
    name: str,
    detect_runs=None,
    n_frames: int | None = None,
    *,
    val_split: float = 0.15,
    score_cutoff: float | None = None,
    exclude_audit_s: float = 1.0,
    every_n_frames: int = 1,
    student_input: str = "whole_frame",
    student_crop_px: int = 448,
    example_set: str | None = None,
    tracking=None,
    model_id: str = "dinov3_match",
    model_variant: str | None = None,
    label_compute=None,
    settings: dict | None = None,
    teacher: dict | None = None,
    class_example_sets: dict[str, str] | None = None,
):
    """Build a YOLO-pose dataset (the label stage of a keypoint distill, usable on its own).
    Returns a ``DatasetRun`` handle; the dataset lands in ``datasets/<id>/`` with
    ``task: pose``.

    The teacher is ``detect_runs`` (recording names or ids, each with a detect run id or
    ``None`` for the recording's only complete one; ``tracking`` may name the tracking run each
    recording's poses are assigned to), or the patch method: ``example_set`` (an
    example set id) with ``tracking`` (recording names or ids, each with a tracking run id or
    ``None``), ``model_id`` / ``model_variant``, ``label_compute`` and ``settings`` (the patch
    method's ``decode`` / ``side_rule`` / ``crop_px``, as ``keypoint_test`` takes them, recorded
    with the dataset); ``class_example_sets`` stands in for ``example_set`` with one set per
    detection class (§26). See
    :func:`plan_keypoint_dataset` and ``KeypointDatasetConfig`` for the rest.
    """
    from detail.api.experiment import DatasetRun, _PIPELINE_RUNNERS
    from detail.executor.handles import ExecutorRunHandle
    from detail.executor.local import LocalExecutor
    from detail.executor.runners import import_runner_modules

    import_runner_modules(_PIPELINE_RUNNERS)
    cfg, teacher, cutoff, compute = plan_keypoint_dataset(
        experiment,
        name,
        detect_runs,
        n_frames,
        val_split=val_split,
        score_cutoff=score_cutoff,
        exclude_audit_s=exclude_audit_s,
        every_n_frames=every_n_frames,
        student_input=student_input,
        student_crop_px=student_crop_px,
        example_set=example_set,
        tracking=tracking,
        model_id=model_id,
        model_variant=model_variant,
        label_compute=label_compute,
        settings=settings,
        teacher=teacher,
        class_example_sets=class_example_sets,
    )
    by_model = teacher["kind"] == "model"
    dataset_id = datasets_store.next_dataset_id(experiment.dir)
    job = KeypointDatasetJob(
        job_id=dataset_id,
        dataset_id=dataset_id,
        name=str(name).strip(),
        teacher=dict(teacher),
        compute=compute,
        n_frames=cfg.n_frames,
        val_split=cfg.val_split,
        score_cutoff=cutoff,
        exclude_audit_s=cfg.exclude_audit_s,
        every_n_frames=cfg.every_n_frames,
        created_at=utc_timestamp(),
        student_input=cfg.student_input,
        student_crop_px=cfg.student_crop_px,
    )
    ddir = datasets_store.dataset_dir(experiment.dir, dataset_id)
    ddir.mkdir(parents=True, exist_ok=True)
    # An initial record so the dataset is discoverable while it runs; the runner's completion
    # write is the full record and always lands last.
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": dataset_id,
            "name": job.name,
            "task": "pose",
            "source": (
                {
                    "kind": "model",
                    "model_id": teacher["model_id"],
                    "tracking_runs": dict(teacher["tracking_runs"]),
                }
                if by_model
                else {
                    "kind": "keypoint_detect",
                    "detect_runs": dict(teacher["detect_runs"]),
                    "tracking_runs": dict(teacher["tracking_runs"]),
                }
            ),
            "skeleton_ref": teacher.get("skeleton_ref") if by_model else None,
            "status": "running",
        },
    )
    executor = LocalExecutor(experiment.dir, runner_modules=_PIPELINE_RUNNERS)
    executor.submit(job)
    return DatasetRun(ExecutorRunHandle(executor, dataset_id), experiment, dataset_id)


def check_pose_base(base_model: str) -> None:
    """A keypoint distill trains from a ``-pose`` checkpoint; anything else cannot learn its
    labels (the trainer re-checks against the dataset once it exists)."""
    stem = Path(str(base_model)).stem.lower()
    if "-pose" not in stem:
        raise ValueError(
            f"base model {base_model!r} is not a pose checkpoint; a keypoint distill trains "
            f"from a -pose base (e.g. {DEFAULT_POSE_BASE})"
        )


def distill_keypoints(
    experiment: "Experiment",
    name: str,
    detect_runs=None,
    n_frames: int | None = None,
    *,
    val_split: float = 0.15,
    score_cutoff: float | None = None,
    exclude_audit_s: float = 1.0,
    every_n_frames: int = 1,
    student_input: str = "whole_frame",
    student_crop_px: int = 448,
    example_set: str | None = None,
    tracking=None,
    model_id: str = "dinov3_match",
    model_variant: str | None = None,
    label_compute=None,
    settings: dict | None = None,
    base_model: str = DEFAULT_POSE_BASE,
    trainer_id: str = "ultralytics_yolo",
    hyperparams: dict | None = None,
    train_compute=None,
    class_example_sets: dict[str, str] | None = None,
):
    """Distill a teacher into a fast YOLO-pose student in one durable run (§12): label →
    train → register. The teacher is ``detect_runs`` or the patch method (``example_set`` —
    or ``class_example_sets``, one set per detection class — with ``tracking``), exactly as
    :func:`keypoint_dataset` takes it. Validates both halves up front, reserves the record and spawns
    the detached distill coordinator; returns an ``AttachedDistillRun``.

    Raises:
        ValueError: Anything :func:`plan_keypoint_dataset` refuses, a base model that is not a
            pose checkpoint, an unknown or unready trainer, an unsupported train backend, or
            invalid hyperparameters.
    """
    from detail.api.experiment import _PIPELINE_RUNNERS
    from detail.executor.local import spawn_coordinator
    from detail.executor.runners import import_runner_modules

    import_runner_modules(_PIPELINE_RUNNERS)
    cfg, teacher, cutoff, compute = plan_keypoint_dataset(
        experiment,
        name,
        detect_runs,
        n_frames,
        val_split=val_split,
        score_cutoff=score_cutoff,
        exclude_audit_s=exclude_audit_s,
        every_n_frames=every_n_frames,
        student_input=student_input,
        student_crop_px=student_crop_px,
        example_set=example_set,
        tracking=tracking,
        model_id=model_id,
        model_variant=model_variant,
        label_compute=label_compute,
        settings=settings,
        class_example_sets=class_example_sets,
    )
    by_model = teacher["kind"] == "model"
    train_compute, hyperparams = student_preflight(
        base_model, trainer_id, train_compute, hyperparams
    )
    run_id = distill_store.next_distill_id(experiment.dir)
    config = {
        "name": str(name).strip(),
        "base_model": base_model,
        "trainer_id": trainer_id,
        # The teacher as resolved at launch (the detect runs; or the patch method's
        # example-set snapshot, settings and tracking runs): the label stage replays it,
        # never the live set (§12).
        "teacher": ({**teacher, "compute": compute.to_dict()} if by_model else teacher),
        "keypoint_dataset": {**cfg.model_dump(), "score_cutoff": cutoff},
        "training": hyperparams,
        "train_compute": train_compute.to_dict(),
    }
    distill_store.write_run(
        experiment.dir,
        run_id,
        {
            "id": run_id,
            "name": config["name"],
            "kind": KIND,
            "status": "running",
            "created_at": utc_timestamp(),
            "config": config,
            "stages": {
                "label": {"status": "pending", "dataset_id": None},
                "train": {"status": "pending", "model_id": None},
                "register": {"status": "pending"},
            },
            "resume": False,
        },
    )
    try:
        pid = spawn_coordinator(
            distill_store.run_dir(experiment.dir, run_id),
            tuple(_PIPELINE_RUNNERS),
            entrypoint="detail.executor.distill_coordinator",
        )
    except Exception as exc:  # noqa: BLE001 — re-raised; must not leave it "running"
        experiment._fail_distill_record(run_id, f"coordinator failed to launch: {exc}")
        raise
    distill_store.set_coordinator_pid(experiment.dir, run_id, pid)
    return experiment.attach_distill_run(run_id)


def student_preflight(
    base_model: str, trainer_id: str, train_compute, hyperparams: dict | None
):
    """``(train_compute, hyperparams)`` for a student's training stage, checked up front: a
    ``-pose`` base, a known trainer installed here (unless it trains on a cluster), a backend
    it supports, a registered remote target. Shared by ``distill_keypoints`` and a keypoint
    setup, so the two refuse the same things in the same words.

    Raises:
        ValueError: the first thing that would fail at the train stage.
    """
    check_pose_base(base_model)
    trainer_cls = TRAINERS.get(trainer_id)
    if trainer_cls is None:
        raise ValueError(
            f"unknown trainer_id {trainer_id!r}; available: {sorted(TRAINERS)}"
        )
    trainer = trainer_cls()
    train_compute = train_compute or default_compute(trainer.supported_backends)
    # a remote job trains on the cluster's install, never this machine's (§6)
    if train_compute.backend != "remote" and not trainer.is_ready():
        raise ValueError(f"trainer {trainer_id!r} is not ready (install its backend)")
    if effective_backend(train_compute) not in trainer.supported_backends:
        raise ValueError(
            f"trainer {trainer_id!r} does not support backend {train_compute.backend!r}; "
            f"supports {trainer.supported_backends}"
        )
    if train_compute.backend == "cuda" and train_compute.device_indices:
        check_cuda_indices(train_compute.device_indices)
    _require_registered_target(train_compute)
    return train_compute, trainer.validate_hyperparams(hyperparams or {})


def _require_registered_target(compute) -> None:
    """A remote target names a registered cluster (Tenet 10: refused before anything runs)."""
    if compute is None or compute.backend != "remote":
        return
    from detail.storage import remote_targets

    if not compute.remote_target:
        raise ValueError(
            "a remote compute target names no remote_target; pick a registered target "
            "(see `detail remote list`)"
        )
    try:
        remote_targets.get_remote_target(compute.remote_target)
    except KeyError as e:
        raise ValueError(str(e)) from e


# -- §26 / #717: keypoint setup — a teacher across acquisition setups, then its student -- #
SETUP_KIND = "keypoint_setup"
SETUP_STAGES = ("detect", "track", "examples", "label", "train", "register")


def keypoint_teacher_plan(experiment: "Experiment", window_s: float = 300.0) -> list:
    """One proposed teacher recording per acquisition setup, for a keypoint setup.

    Each recording with video is grouped by its ``acquisition_setup`` (§11); the first in
    recording order represents its setup. A recording longer than ``window_s`` is marked
    ``windowed``: the teacher reads only its first ``window_s`` seconds.

    Returns:
        One dict per setup, in the order setups first appear: ``acquisition_setup``,
        ``recording_id``, ``name``, ``duration_s``, ``windowed``, ``recordings`` (every
        recording of that setup, for a surface that lets the person swap the pick) and
        ``durations`` (each of those recordings' length in seconds).

    Raises:
        ValueError: no recording has video, or a recording with video has no
            ``acquisition_setup`` — named, since a guess would leave a setup out of the
            teacher.
    """
    from detail.storage import metadata as metadata_store

    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    table = experiment.recording_metadata
    setups: dict[str, list] = {}
    blank = []
    for recording in experiment.recordings:
        data = recording_store.read_recording(recording.dir)
        video = data.get("video") or {}
        if not video.get("files"):
            continue
        setup = (
            str(
                (table.get(recording.id) or {}).get(
                    metadata_store.ACQUISITION_SETUP_KEY, ""
                )
                or ""
            )
        ).strip()
        if not setup:
            blank.append(recording.name)
            continue
        setups.setdefault(setup, []).append(
            (recording, float(video.get("total_duration_s") or 0.0))
        )
    if blank:
        shown = ", ".join(repr(n) for n in blank[:10])
        more = f" and {len(blank) - 10} more" if len(blank) > 10 else ""
        raise ValueError(
            f"{len(blank)} recording(s) have no acquisition_setup: {shown}{more}. "
            "Fill it in on Recording information (the same value for every recording "
            "made with the same camera or box), so the teacher sees every setup."
        )
    if not setups:
        raise ValueError("no recording has video; import video before keypoint setup")
    return [
        {
            "acquisition_setup": setup,
            "recording_id": members[0][0].id,
            "name": members[0][0].name,
            "duration_s": members[0][1],
            "windowed": members[0][1] > window_s,
            "recordings": [r.id for r, _ in members],
            "durations": {r.id: duration for r, duration in members},
        }
        for setup, members in setups.items()
    ]


def _setup_cutoff(model_cls, score_cutoff: float | None) -> float | None:
    """The cut-off a setup teaches with: the one given, else the method's own default."""
    if score_cutoff is not None:
        return float(score_cutoff)
    spec = getattr(model_cls, "score_spec", None)
    cutoff = getattr(spec, "default_cutoff", None)
    return None if cutoff is None else float(cutoff)


def _check_example_set(experiment: "Experiment", example_set: str):
    """The set a setup's teacher learns from, ready to run (``purpose: example``, every
    point clicked visible on some frame)."""
    try:
        examples = example_sets.read_set(experiment.dir, example_set)
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from None
    if getattr(examples, "purpose", "example") != "example":
        raise ValueError(
            f"{example_set!r} is an audit set; a teacher learns from an example set"
        )
    examples.check_runnable()
    return examples


def _setup_classes(
    class_name: str | None,
    class_names: list[str] | None,
    example_set: str | None,
    class_example_sets: dict[str, str] | None,
) -> list[str]:
    """A setup's animal classes, checked: ``[class_name]``, or ``class_names`` (each tracked
    on its own, then merged; a one-class list is the one-animal setup). Two or more animals
    learn from one set per class, never one set for all.

    Raises:
        ValueError: Both or neither of ``class_name`` / ``class_names``, a class named twice,
            ``example_set`` with several animals, a per-class map that does not name each
            animal, or a per-class map for one animal.
    """
    if (class_name is None) == (not class_names):
        raise ValueError(
            "name the animal: class_name for one, or class_names for two or more (each "
            "tracked on its own, then merged)"
        )
    animals = [class_name] if class_name is not None else list(class_names)
    if len(set(animals)) != len(animals):
        raise ValueError(f"class_names names a class twice: {animals}")
    if len(animals) > 1:
        if example_set is not None:
            raise ValueError(
                "two or more animals learn from one example set per class: give "
                "class_example_sets ({class: set}), not example_set"
            )
        if class_example_sets and set(class_example_sets) != set(animals):
            raise ValueError(
                f"class_example_sets must name each of {sorted(animals)} once; got "
                f"{sorted(class_example_sets)}"
            )
    elif class_example_sets:
        raise ValueError(
            "class_example_sets is for two or more animals; give the one animal's "
            "example_set"
        )
    return animals


def _check_class_example_sets(
    experiment: "Experiment", animals: list[str], class_example_sets: dict[str, str]
) -> None:
    """Each animal's set ready to run, and every set on one skeleton (one run stores one set
    of points, §26).

    Raises:
        ValueError: A set not ready to run, or sets on different skeletons.
    """
    from detail.pipeline.keypoint_detect import _raise_on_mixed_skeletons

    by_class = {
        cls: _check_example_set(experiment, class_example_sets[cls]) for cls in animals
    }
    _raise_on_mixed_skeletons(by_class)


def start_keypoint_setup(
    experiment: "Experiment",
    name: str,
    detector_model_id: str,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
    class_example_sets: dict[str, str] | None = None,
    teachers: dict | None = None,
    window_s: float = 300.0,
    conf_threshold: float = 0.5,
    tracker_id: str | None = None,
    detect_compute=None,
    keypoints_model_id: str = "dinov3_match",
    keypoints_model_variant: str | None = None,
    keypoint_settings: dict | None = None,
    keypoints_compute=None,
    example_set: str | None = None,
    n_frames: int = 3000,
    score_cutoff: float | None = None,
    val_split: float = 0.15,
    exclude_audit_s: float = 1.0,
    every_n_frames: int = 1,
    student_input: str = "whole_frame",
    student_crop_px: int = 448,
    base_model: str = DEFAULT_POSE_BASE,
    trainer_id: str = "ultralytics_yolo",
    hyperparams: dict | None = None,
    train_compute=None,
):
    """Build a keypoint teacher across the experiment's acquisition setups, then distill it
    into a fast student: one durable distill run of kind ``keypoint_setup`` (§26, #717).

    One recording per setup (:func:`keypoint_teacher_plan`; ``teachers`` maps a setup to
    another of its recordings), each limited to its first ``window_s`` seconds. The run
    detects ``class_name`` with ``detector_model_id`` (a distilled, closed-set detector),
    tracks it, and stops at ``awaiting_references``: click an example set on those tracks
    (their ids are the record's ``stages.track.run_ids``), then
    :func:`continue_keypoint_setup`. Given an ``example_set`` already clicked (on any
    recordings of this animal) it does not stop. The teacher is then the patch method
    (``keypoints_model_id`` with that set, run on the sampled frames of the teacher tracks)
    and the rest is a keypoint distill (:func:`distill_keypoints`), whose parameters the
    remaining keywords are.

    **Two or more animals** (§26 *One example set per detection class*): ``class_names``, in
    place of ``class_name``, tracks each class on its own, merges the tracks per teacher
    recording (``stages.merge.run_ids`` are the teacher tracks) and waits for one example set
    per class — or goes straight through given ``class_example_sets`` (``{class: set}``,
    in place of ``example_set``). One student is trained for every class.

    Raises:
        ValueError: anything that would fail later, checked now (Tenet 10): no name, a blank
            or unknown setup, a teacher not of its setup, a detector that is open-set or
            lacks a class asked for, both or neither of ``class_name`` / ``class_names``, a
            keypoint method that learns from no example set, an example set not ready to
            run, a per-class map that does not name each class once or mixes skeletons, an
            unknown tracker, an unsupported backend or an unregistered target, no score
            cut-off where one is needed, or a bad dataset or training parameter.
    """
    from detail.api.experiment import _PIPELINE_RUNNERS
    from detail.core.registries import TRACKERS
    from detail.executor.runners import import_runner_modules
    from detail.models import resolve as models_resolve

    import_runner_modules(_PIPELINE_RUNNERS)
    if not name or not str(name).strip():
        raise ValueError("give the keypoint setup a name")
    plan = keypoint_teacher_plan(experiment, window_s)
    by_setup = {row["acquisition_setup"]: row for row in plan}
    for setup, ref in (teachers or {}).items():
        if setup not in by_setup:
            raise ValueError(
                f"no acquisition setup {setup!r}; this experiment has {sorted(by_setup)}"
            )
        recording_id = experiment.resolve_recording(ref).id
        if recording_id not in by_setup[setup]["recordings"]:
            raise ValueError(
                f"recording {ref!r} is not one of setup {setup!r}'s recordings"
            )
        by_setup[setup] = {**by_setup[setup], "recording_id": recording_id}
    recording_ids = [row["recording_id"] for row in by_setup.values()]

    detector, _ = models_resolve.resolve_detection_model_auto(
        experiment.dir, detector_model_id
    )
    if detector.supported_cue_types:
        raise ValueError(
            f"{detector_model_id!r} is a prompted model; keypoint setup detects with a "
            "distilled detector (distill a model first, then pick it here)"
        )
    classes = list(getattr(detector, "classes", []))
    refs = experiment.resolve_class_refs(
        {
            "class_name": class_name,
            "class_names": class_names,
            "class_example_sets": class_example_sets,
        }
    )
    class_name, class_names = refs["class_name"], refs["class_names"]
    class_example_sets = refs["class_example_sets"]
    animals = _setup_classes(class_name, class_names, example_set, class_example_sets)
    for animal in animals:
        if animal not in classes:
            raise ValueError(
                f"detector {detector_model_id!r} has no class {animal!r}; "
                f"it detects {classes}"
            )
    detect_compute = detect_compute or default_compute(detector.supported_backends)
    if effective_backend(detect_compute) not in detector.supported_backends:
        raise ValueError(
            f"detector {detector_model_id!r} does not support the "
            f"{detect_compute.backend!r} backend; supports {detector.supported_backends}"
        )
    _require_registered_target(detect_compute)
    # Resolved now and recorded, so the run names the tracker its track stage used (Tenet 9).
    tracker_id = tracker_id or DEFAULT_TRACKER
    if tracker_id not in TRACKERS:
        raise ValueError(
            f"unknown tracker_id {tracker_id!r}; available: {sorted(TRACKERS)}"
        )
    method = models_resolve.keypoint_method_class(experiment.dir, keypoints_model_id)
    if method is None or not getattr(method, "needs_examples", False):
        learners = sorted(
            m for m, c in KEYPOINT_MODELS.items() if getattr(c, "needs_examples", False)
        )
        raise ValueError(
            f"{keypoints_model_id!r} places points without an example set, so it cannot "
            f"be taught this experiment's animal; keypoint setup needs one of {learners}"
        )
    method_backends = list(getattr(method, "supported_backends", ["cpu", "cuda"]))
    keypoints_compute = keypoints_compute or default_compute(method_backends)
    if effective_backend(keypoints_compute) not in method_backends:
        raise ValueError(
            f"{keypoints_model_id!r} does not support the "
            f"{keypoints_compute.backend!r} backend; supports {method_backends}"
        )
    _require_registered_target(keypoints_compute)
    cutoff = _setup_cutoff(method, score_cutoff)
    if example_set is not None or class_example_sets:
        if class_example_sets:
            _check_class_example_sets(experiment, animals, class_example_sets)
        else:
            _check_example_set(experiment, example_set)
        if cutoff is None:
            raise ValueError(
                f"{keypoints_model_id} sets no default score cut-off; pass score_cutoff "
                "(Test cut-off… on the teacher tracks shows where to put it), or leave "
                "the example set out and choose both when the setup waits for them"
            )
    kp_cfg = KeypointDatasetConfig.model_validate(
        {
            "n_frames": n_frames,
            "val_split": val_split,
            "score_cutoff": cutoff,
            "exclude_audit_s": exclude_audit_s,
            "every_n_frames": every_n_frames,
            "student_input": student_input,
            "student_crop_px": student_crop_px,
        }
    )
    train_compute, hyperparams = student_preflight(
        base_model, trainer_id, train_compute, hyperparams
    )
    run_id = distill_store.next_distill_id(experiment.dir)
    config = {
        "kind": SETUP_KIND,
        "name": str(name).strip(),
        "recording_ids": recording_ids,
        "teachers": {setup: row["recording_id"] for setup, row in by_setup.items()},
        "window_s": float(window_s),
        "detector_model_id": detector_model_id,
        # one animal (class_name), or two or more each tracked and then merged (class_names)
        "class_name": None if len(animals) > 1 else animals[0],
        "class_names": list(animals) if len(animals) > 1 else None,
        "conf_threshold": float(conf_threshold),
        "tracker_id": tracker_id,
        "detect_compute": detect_compute.to_dict(),
        "keypoints_model_id": keypoints_model_id,
        "keypoints_model_variant": keypoints_model_variant,
        "keypoint_settings": dict(keypoint_settings or {}) or None,
        "keypoints_compute": keypoints_compute.to_dict(),
        "example_set_id": example_set,
        "class_example_sets": (
            dict(class_example_sets)
            if class_example_sets and len(animals) > 1
            else None
        ),
        # the patch teacher, resolved by the coordinator once the tracks exist
        "teacher": None,
        "base_model": base_model,
        "trainer_id": trainer_id,
        "keypoint_dataset": kp_cfg.model_dump(),
        "training": hyperparams,
        "train_compute": train_compute.to_dict(),
    }
    teacher_stages = (
        ("detect", "track", "merge", "examples")
        if len(animals) > 1
        else ("detect", "track", "examples")
    )
    stages = {stage: {"status": "pending"} for stage in teacher_stages}
    stages.update(
        {
            "label": {"status": "pending", "dataset_id": None},
            "train": {"status": "pending", "model_id": None},
            "register": {"status": "pending"},
        }
    )
    distill_store.write_run(
        experiment.dir,
        run_id,
        {
            "id": run_id,
            "name": config["name"],
            "kind": SETUP_KIND,
            "status": "running",
            "created_at": utc_timestamp(),
            "config": config,
            "stages": stages,
            "resume": False,
        },
    )
    return experiment._spawn_distill_coordinator(run_id)


def continue_keypoint_setup(
    experiment: "Experiment",
    run_id: str,
    example_set: str | None = None,
    score_cutoff: float | None = None,
    class_example_sets: dict[str, str] | None = None,
):
    """Carry a keypoint setup on past its pause: teach with ``example_set`` (clicked on the
    teacher tracks, or on any recordings of this animal), then distill (§26). Reuses the
    finished detection and tracking. ``score_cutoff`` replaces the one given at start. A
    setup with several animals takes ``class_example_sets`` (``{class: set}``, each of its
    ``class_names`` once) instead.

    Raises:
        FileNotFoundError: no such distill run.
        ValueError: the run is not waiting for examples, the set (or a class's set) is not
            ready to run, the wrong form for this setup's animals, or there is no score
            cut-off and the method sets none.
    """
    from detail.api.experiment import setup_class_names

    from detail.api.experiment import _PIPELINE_RUNNERS
    from detail.core.dataclasses import AWAITING_REFERENCES
    from detail.executor.runners import import_runner_modules
    from detail.models import resolve as models_resolve

    import_runner_modules(_PIPELINE_RUNNERS)
    record = distill_store.read_run(experiment.dir, run_id)
    if record.get("status") != AWAITING_REFERENCES:
        raise ValueError(
            f"distill run {run_id!r} is {record.get('status')!r}; only a keypoint setup "
            "waiting for its example set can be continued"
        )
    animals = setup_class_names(record)
    class_example_sets = experiment.resolve_class_refs(
        {"class_example_sets": class_example_sets}
    )["class_example_sets"]
    if animals:
        if example_set is not None or not class_example_sets:
            raise ValueError(
                f"this setup tracks {animals}: continue it with one example set per class "
                "(class_example_sets)"
            )
        _setup_classes(None, animals, None, class_example_sets)
        _check_class_example_sets(experiment, animals, class_example_sets)
    else:
        if class_example_sets or example_set is None:
            raise ValueError(
                "this setup tracks one animal: continue it with its example_set"
            )
        _check_example_set(experiment, example_set)
    cfg = record["config"]
    method = models_resolve.keypoint_method_class(
        experiment.dir, cfg["keypoints_model_id"]
    )
    cutoff = _setup_cutoff(
        method,
        (
            score_cutoff
            if score_cutoff is not None
            else (cfg.get("keypoint_dataset") or {}).get("score_cutoff")
        ),
    )
    if cutoff is None:
        raise ValueError(
            f"{cfg['keypoints_model_id']} sets no default score cut-off; pass score_cutoff "
            "(Test cut-off… on the teacher tracks shows where to put it)"
        )
    cfg["example_set_id"] = example_set
    cfg["class_example_sets"] = dict(class_example_sets) if animals else None
    cfg["teacher"] = None  # resolved again, from this set
    cfg["keypoint_dataset"] = {
        **(cfg.get("keypoint_dataset") or {}),
        "score_cutoff": cutoff,
    }
    record["status"] = "running"
    record["resume"] = True
    record.pop("error", None)
    distill_store.write_run(experiment.dir, run_id, record)
    distill_store.clear_cancel(experiment.dir, run_id)
    return experiment._spawn_distill_coordinator(run_id)


def register_student(experiment: "Experiment", run_id: str, record: dict) -> dict:
    """The register stage: promote the trained student into the experiment's catalog as
    ``<name> <distill_NNN>`` with ``role: keypoints``, the weights' sha256 and the teacher
    runs. Idempotent, so a resume after a failed register does not retrain.

    Raises:
        ValueError: The train stage recorded no model, or the model has no best weights.
    """
    stages = record.get("stages") or {}
    model_id = (stages.get("train") or {}).get("model_id")
    if not model_id:
        raise ValueError("the train stage recorded no model to register")
    model_dir = models_store.model_dir(experiment.dir, model_id)
    model = models_store.read_model_yaml(model_dir)
    best = (model.get("weights") or {}).get("best")
    if not best or not (model_dir / best).is_file():
        raise ValueError(f"model {model_id!r} has no best weights to register")
    digest = hashlib.sha256((model_dir / best).read_bytes()).hexdigest()
    cfg = record.get("config") or {}
    model["name"] = f"{cfg.get('name') or model.get('name') or model_id} {run_id}"
    model["role"] = "keypoints"
    model["task"] = "pose"
    model["model_family"] = "yolo_pose"
    model["weights_sha256"] = digest
    # A record written before the teacher field names track runs (the retired teacher).
    teacher = cfg.get("teacher") or {
        "kind": "keypoint_track",
        "track_runs": dict(cfg.get("track_runs") or {}),
    }
    model["teacher"] = {
        "kind": teacher.get("kind"),
        "distill_run_id": run_id,
        "dataset_id": (stages.get("label") or {}).get("dataset_id"),
    }
    for runs in ("detect_runs", "track_runs"):
        if teacher.get(runs):
            model["teacher"][runs] = dict(teacher[runs])
    if teacher.get("kind") == "keypoint_detect" and teacher.get("tracking_runs"):
        # the subjects the detect runs' poses were assigned to (§26 *Assignment*)
        model["teacher"]["tracking_runs"] = dict(teacher["tracking_runs"])
    if teacher.get("kind") == "model":
        model["teacher"].update(
            model_id=teacher.get("model_id"),
            example_set={
                "id": (teacher.get("example_set") or {}).get("id"),
                "digest": (teacher.get("example_set") or {}).get("digest"),
            },
            tracking_runs=dict(teacher.get("tracking_runs") or {}),
        )
        if teacher.get("example_sets"):
            # A per-class teacher (§26): which set each class was matched against.
            model["teacher"]["example_sets"] = {
                cls: {"id": snap.get("id"), "digest": snap.get("digest")}
                for cls, snap in teacher["example_sets"].items()
            }
        if teacher.get("skeleton_ref"):
            model["skeleton_ref"] = dict(teacher["skeleton_ref"])
    if not model.get("skeleton"):
        dataset_id = (stages.get("label") or {}).get("dataset_id")
        if dataset_id:
            dataset = datasets_store.read_dataset_yaml(
                datasets_store.dataset_dir(experiment.dir, dataset_id)
            )
            if dataset.get("skeleton"):
                model["skeleton"] = dataset["skeleton"]
    models_store.write_model_yaml(model_dir, model)
    return model
