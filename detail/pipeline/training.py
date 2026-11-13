"""Training runner — train a custom detection model from a dataset (§12).

A long GPU job: resolve the trainer (TRAINERS registry) + dataset, call
``trainer.train`` (which writes weights into ``models/<id>/run/``), then record
the trained-weight paths (relative — Tenet 6), the dataset's classes, the
trainer's metrics, and what it trained with (``hyperparams`` + the backend's resolved
``trained_with``) into ``model.yaml``. The trainer backend (e.g. Ultralytics) is
lazy-imported inside its own class (Tenet 13); this module imports none.
"""

import contextlib
from pathlib import Path

from detail.core.dataclasses import TrainingJob
from detail.core.keypoints import student_input_of
from detail.core.registries import TRAINERS
from detail.core.utils import resolve_device
from detail.executor import rundir
from detail.executor.runners import register_runner
from detail.models.trained_yolo import TrainedYOLO
from detail.storage import datasets as datasets_store
from detail.storage import models as models_store
from detail.storage import experiment as experiment_store


def run_dir_for(experiment_root, job: TrainingJob) -> Path:
    """``models/<model_id>/`` — a experiment-level run dir (not per-recording)."""
    return experiment_store.models_dir(experiment_root) / job.model_id


@register_runner("training", run_dir_for=run_dir_for)
def run(job: TrainingJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)  # models/<model_id>/
    try:
        record = _train(job, run_dir, progress_cb, stop_event)
    except Exception as exc:
        # The model's status is authoritative for every reader (not only callers
        # that reconcile via ModelRun.wait), so a crashed train must not be left
        # stuck at "running" — that would hide it from delete (gui_design §1.1).
        # The message is recorded too so the GUI can show why it failed.
        with contextlib.suppress(FileNotFoundError):
            models_store.set_model_status(
                run_dir, "failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise
    # Outside the `try` on purpose. This write does not *do* the work, it **records that the
    # work succeeded** — so an `except` that stamps `failed` must not span it. It used to be
    # the last statement of `_train`, inside the try, which meant a transient error on a
    # network share marked hours of finished training `failed` with `best.pt` sitting on disk
    # and unreachable from the catalog. The recorded class: an isolation boundary may not span
    # the bookkeeping about the work it isolates.
    models_store.write_model_yaml(run_dir, record)


def _train(job: TrainingJob, run_dir: Path, progress_cb, stop_event) -> dict:
    """Train, and **return** the record describing what was produced — never write it.

    The caller writes it, outside the handler that stamps ``failed``: the write is a
    statement *about* the work, so putting it inside that handler made a transient file error
    a verdict on the training.
    """
    experiment_root = run_dir.parent.parent
    trainer_cls = TRAINERS.get(job.trainer_id)
    if trainer_cls is None:
        raise ValueError(
            f"unknown trainer_id {job.trainer_id!r}; available: {sorted(TRAINERS)}"
        )
    trainer = trainer_cls()
    dataset_dir = datasets_store.dataset_dir(experiment_root, job.dataset_id)
    dataset = datasets_store.read_dataset_yaml(dataset_dir)
    device = resolve_device(job.compute, trainer.supported_backends)

    trainer.prepare_dataset(dataset_dir, run_dir)
    # Resume from the last checkpoint if a prior (interrupted) run left one; otherwise train
    # from scratch. A fresh model_id has an empty run dir, so the first run trains;
    # ``Experiment.resume_training`` re-submits the same id (``last.pt`` present) → this resumes.
    if trainer.checkpoint_path(run_dir) is not None:
        trainer.resume(job, dataset_dir, run_dir, device, progress_cb, stop_event)
    else:
        trainer.train(job, dataset_dir, run_dir, device, progress_cb, stop_event)

    # The stop event means only "stop now". Which marker set it says what the user wanted: a
    # stop-and-keep is a finish they chose (→ complete), a cancel is an interruption
    # (→ partial). Ultralytics finalizes the run on either — best weights kept, `last.pt`
    # stripped of its resume state — so a stopped model is finished, not resumable (#486).
    # A stop with no stop-and-keep marker behind it keeps the conservative reading.
    stopped = stop_event is not None and stop_event.is_set()
    stopped_early = (
        stopped
        and rundir.stop_requested(run_dir)
        and not rundir.cancel_requested(run_dir)
    )
    interrupted = stopped and not stopped_early
    best = trainer.best_weights_path(run_dir)
    last = trainer.checkpoint_path(run_dir)
    if best is None and not interrupted:
        raise RuntimeError("training finished but produced no best-weights file")

    # The API wrote an initial model.yaml (id/name/trainer/dataset/base_model);
    # fill in what the run produced, keeping weight paths relative (Tenet 6).
    record = models_store.read_model_yaml(run_dir)
    record["model_family"] = trainer.model_family  # resolves the trained-apply loader
    record["classes"] = dataset.get("classes", [])
    if dataset.get("task"):
        # A pose dataset (§12 *Distill keypoints*): the student carries the task and the
        # skeleton copy it was trained on, which is what the keypoint resolver reads back.
        record["task"] = dataset["task"]
        record["model_family"] = (
            "yolo_pose" if dataset["task"] == "pose" else trainer.model_family
        )
        if dataset.get("skeleton"):
            record["skeleton"] = dataset["skeleton"]
        if dataset["task"] == "pose":
            # how the student must be run: on the whole frame, or a crop per animal (§12)
            # read as the loader reads it (core.keypoints.student_input_of): a pre-§26
            # dataset recorded no student_input, only its crop size, under ``source``, and
            # was a crop dataset
            source_crop = (dataset.get("source") or {}).get("crop_px")
            student = student_input_of({"crop_px": source_crop, **dataset})
            record["student_input"] = student["student_input"]
            record["student_crop_px"] = (
                student["crop_px"]
                if student["student_input"] == "crop"
                else dataset.get("student_crop_px")
            )
    record["weights"] = {
        "best": best.relative_to(run_dir).as_posix() if best is not None else None,
        "last": last.relative_to(run_dir).as_posix() if last is not None else None,
    }
    record["metrics"] = trainer.final_metrics(run_dir) if best is not None else {}
    # What trained it: the settings asked for (also written at submit, so a live page has the
    # epoch count — restated here for a model submitted before that), and every setting the
    # backend resolved, with its version.
    record["hyperparams"] = dict(job.hyperparams)
    record["trained_with"] = trainer.resolved_settings(run_dir)
    record["preview"] = (
        _capture_preview(
            dataset_dir,
            best,
            record["classes"],
            run_dir,
            device,
            progress_cb,
            int(job.hyperparams.get("epochs") or 0),
        )
        if best is not None and not interrupted
        else None
    )
    record["status"] = "partial" if interrupted else "complete"
    if stopped_early:
        # Kept beside `complete`, never instead of it: a model stopped at epoch 5 of 100 is
        # finished, and must still not read as one that ran its course.
        record["stopped_at_epoch"] = record["metrics"].get("final_epoch")
    else:
        record.pop("stopped_at_epoch", None)
    return record


_PREVIEW_CONF = 0.25  # keep the thumbnail from filling with low-confidence boxes


def _capture_preview(
    dataset_dir: Path,
    best: Path,
    classes: list,
    run_dir: Path,
    device: str,
    progress_cb,
    total_epochs: int = 0,
) -> str | None:
    """Render one overlay frame of the model's own prediction as its preview thumbnail
    (§12) — a "what this model does" shot for the catalog. Best-effort at an isolation
    boundary: the preview is cosmetic, so *any* failure yields no preview and never fails
    an otherwise-valid trained model. Sources the frame from a dataset val JPG (no video
    machinery), runs one closed-set inference, and reuses the prompt-test overlay."""
    try:
        import cv2

        from detail.core.dataclasses import Prompt, TrainingProgressEvent
        from detail.core.visualize import (
            assign_colors,
            bgr_to_rgb,
            draw_detections,
            rgb_to_bgr,
        )

        val_dir = dataset_dir / "images" / "val"
        frames = sorted(val_dir.glob("*.jpg")) or sorted(val_dir.glob("*.png"))
        # A background frame (empty label) would preview a model finding nothing.
        labels_dir = dataset_dir / "labels" / "val"
        frames = [
            f
            for f in frames
            if (labels_dir / f"{f.stem}.txt").is_file()
            and (labels_dir / f"{f.stem}.txt").read_text(encoding="utf-8-sig").strip()
        ] or frames
        if not frames:
            return None
        frame_bgr = cv2.imread(str(frames[0]))
        if frame_bgr is None:
            return None
        if progress_cb is not None:  # a ~1–3 s tail step — name it, don't sit silent
            # Counted as one unit *past* the epochs rather than as its own 1/1: the
            # renderer maps epoch/total_epochs to the bar, so 1/1 read 100% while this
            # step was still running. N/(N+1) leaves the last slice for the tail.
            progress_cb(
                TrainingProgressEvent(
                    epoch=max(total_epochs, 1),
                    total_epochs=max(total_epochs, 1) + 1,
                    metrics={},
                    elapsed_s=0.0,
                    eta_s=None,
                    message="writing preview",
                )
            )
        model = TrainedYOLO(best, [c["name"] for c in classes])
        model.load(device)
        result = model.predict(
            frame_bgr, Prompt(targets=[])
        )  # closed-set: ignores prompt
        kept = [d for d in result.detections() if d.confidence >= _PREVIEW_CONF]
        color = assign_colors(["subject"], seed=0)["subject"]
        overlay = draw_detections(bgr_to_rgb(frame_bgr), kept, color)
        cv2.imwrite(str(run_dir / "preview.png"), rgb_to_bgr(overlay))
        return "preview.png"
    except Exception:
        # A cosmetic side-artifact must never fail a valid trained model (isolation
        # boundary): skip the preview on any error and record none.
        return None
