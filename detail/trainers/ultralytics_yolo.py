"""Ultralytics YOLO trainer — train a closed-set YOLO-seg detection model (§12).

``from ultralytics import YOLO`` is imported lazily inside the methods (Tenet 13),
so this module imports no backend and registration works even when Ultralytics is
absent. Base weights and training run under ``contextlib.chdir`` into the shared
model cache so ``yolo*.pt`` never lands in the repo.

The trained model is **closed-set** — its classes come from the dataset, and it
takes no prompt cues (``supported_cue_types = []``), per the §12 design decision.
"""

import contextlib
import csv
import hashlib
import importlib.metadata
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from detail.configs.training import UltralyticsYOLOConfig
from detail.core import atomic
from detail.core.dataclasses import TrainingProgressEvent
from detail.core.installed import is_importable
from detail.core.progress import remaining_seconds
from detail.core.protocols import Trainer
from detail.core.registries import register_trainer
from detail.models._ultralytics import import_ultralytics
from detail.models.hf_access import model_cache_dir


def _weights_dir() -> Path:
    """Shared cache for Ultralytics base weights (downloader writes to the cwd)."""
    path = model_cache_dir() / "ultralytics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _installed_version(package: str) -> str | None:
    """The installed distribution's version, or None when it is not installed."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _machine_path(value) -> bool:
    """Whether a resolved setting is a path on this machine (``data``, ``project``,
    ``save_dir``, a resumed ``model``): meaningless anywhere else, and an absolute path may not
    be written into an experiment file."""
    if not isinstance(value, str) or not value:
        return False
    return (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("~")
    )


def _resumable(ckpt) -> bool:
    """Whether an Ultralytics checkpoint dict still carries the state a resume needs.

    Ultralytics strips ``epoch`` (to -1) and ``optimizer`` (to None) when a run finishes, early
    or not; the per-epoch save while it is still going keeps both. A killed or crashed run
    therefore leaves a resumable ``last.pt`` and a finished one does not.
    """
    if not isinstance(ckpt, dict):
        return False
    epoch = ckpt.get("epoch")
    return epoch is not None and epoch >= 0 and ckpt.get("optimizer") is not None


@register_trainer("ultralytics_yolo")
class UltralyticsYOLOTrainer(Trainer):
    trainer_id = "ultralytics_yolo"
    model_family = "yolo"
    supported_cue_types = []  # closed-set — the trained model takes no prompt cues
    supported_backends = ["cpu", "cuda", "mps"]

    config_model = UltralyticsYOLOConfig

    def is_ready(self) -> bool:
        return is_importable("ultralytics")

    def validate_hyperparams(self, hyperparams: dict) -> dict:
        """Fill defaults and enforce bounds via the config model (raises
        ``pydantic.ValidationError``, a ``ValueError``, on an unknown or
        out-of-range key)."""
        return self.config_model.model_validate(hyperparams).model_dump()

    def prepare_dataset(self, dataset_dir: Path, work_dir: Path) -> Path:
        """Our export datasets are already YOLO-format; the descriptor is
        ``data.yaml`` (``path: .``). Validate it exists and return it."""
        data_yaml = Path(dataset_dir) / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"dataset {dataset_dir} has no data.yaml")
        return data_yaml

    def _training_data_yaml(self, dataset_dir: Path, model_dir: Path) -> Path:
        """A copy of the dataset's ``data.yaml`` whose ``path`` is the absolute
        dataset directory.

        The on-disk descriptor keeps ``path: .`` for portability, but Ultralytics
        resolves a relative ``path:`` against the *current working directory* —
        and ``train`` runs under ``chdir`` into the weights cache, so ``.`` would
        resolve there, not at the dataset. The copy is written to the machine-local
        weights cache (never a experiment file) at a stable name keyed by ``model_dir``
        so the path Ultralytics bakes into ``run/args.yaml`` survives for resume.
        """
        spec = yaml.safe_load(
            (Path(dataset_dir) / "data.yaml").read_text(encoding="utf-8-sig")
        )
        spec["path"] = str(Path(dataset_dir).resolve())
        key = hashlib.sha1(str(Path(model_dir).resolve()).encode()).hexdigest()[:16]
        out_dir = _weights_dir() / "data_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{key}.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f, sort_keys=False)
        return out

    def train(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ) -> None:
        import_ultralytics()
        from ultralytics import YOLO

        hp = self.validate_hyperparams(job.hyperparams)
        if (
            progress_cb is not None
        ):  # the base-weights download + spin-up is otherwise silent
            progress_cb(
                TrainingProgressEvent(
                    epoch=0,
                    total_epochs=hp["epochs"],
                    metrics={},
                    elapsed_s=0.0,
                    eta_s=None,
                    message="loading base model",
                )
            )
        self.prepare_dataset(dataset_dir, model_dir)  # validate the descriptor exists
        self.check_base_model(job.base_model, dataset_dir)
        data_yaml = self._training_data_yaml(dataset_dir, model_dir)
        with contextlib.chdir(_weights_dir()):  # base weights download into the cache
            model = YOLO(job.base_model)
        self._wire_callbacks(model, hp["epochs"], progress_cb, stop_event)
        with contextlib.chdir(_weights_dir()):
            model.train(
                data=str(data_yaml),
                epochs=hp["epochs"],
                batch=hp["batch_size"],
                imgsz=hp["imgsz"],
                patience=hp["patience"],
                project=str(model_dir),
                name="run",
                device=device,
                exist_ok=True,
                verbose=False,
            )

    def resume(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ) -> None:
        import_ultralytics()
        from ultralytics import YOLO

        last = self.checkpoint_path(model_dir)
        if last is None:
            raise FileNotFoundError(f"no checkpoint to resume in {model_dir}")
        # Refresh the machine-local data config so a dataset moved on disk resumes against its
        # current location, not the stale path baked into the checkpoint; prepare_dataset raises a
        # clear error if the dataset descriptor is gone. (A experiment moved across machines still
        # can't resume — the checkpoint's args.yaml holds this machine's cache path.)
        self.prepare_dataset(dataset_dir, model_dir)
        self._training_data_yaml(dataset_dir, model_dir)
        model = YOLO(str(last))
        # Refuse a finalized checkpoint here rather than let Ultralytics decide. Given one with
        # no epoch or optimizer state, `train(resume=True)` warns and **starts a new training**
        # built from its own defaults — `coco8-seg.yaml`, batch 16, the global runs dir — since
        # this call passes nothing else (ultralytics 8.4.120 `engine/model.py:813-823`). Seen
        # three times on one machine before #486: every Stop-and-keep model it was offered on.
        if not _resumable(getattr(model, "ckpt", None)):
            raise ValueError(
                f"{last} is a finished training checkpoint, not a resumable one — the run "
                "ended normally (a completed run, or Stop and keep), so there is nothing to "
                "resume. The model's best weights are usable as they are; train a new model "
                "to train further."
            )
        self._wire_callbacks(
            model,
            self.validate_hyperparams(job.hyperparams)["epochs"],
            progress_cb,
            stop_event,
        )
        with contextlib.chdir(_weights_dir()):
            model.train(resume=True)

    @staticmethod
    def dataset_task(dataset_dir: Path | str) -> str:
        """``"pose"`` when the dataset's ``data.yaml`` carries ``kpt_shape`` (§12 *Distill
        keypoints*), else ``"segment"``: what a base checkpoint has to be able to learn.
        """
        spec = yaml.safe_load(atomic.read_text(Path(dataset_dir) / "data.yaml"))
        return "pose" if (spec or {}).get("kpt_shape") else "segment"

    def check_base_model(self, base_model: str, dataset_dir: Path | str) -> None:
        """Refuse a base checkpoint that cannot learn the dataset's labels before anything is
        downloaded or written: a pose dataset needs a ``-pose`` base and a segmentation dataset
        must not have one. Read off the checkpoint's name, Ultralytics' own convention
        (``yolo11s-pose.pt`` / ``yolo11s-seg.pt``), so no backend import is needed here.

        Raises:
            ValueError: The base model's task and the dataset's disagree.
        """
        task = self.dataset_task(dataset_dir)
        stem = Path(str(base_model)).stem.lower()
        base_is_pose = stem.endswith("-pose") or "-pose" in stem
        if task == "pose" and not base_is_pose:
            raise ValueError(
                f"base model {base_model!r} is not a pose checkpoint; a keypoint dataset "
                "trains from a -pose base (e.g. yolo11s-pose.pt)"
            )
        if task != "pose" and base_is_pose:
            raise ValueError(
                f"base model {base_model!r} is a pose checkpoint, but the dataset has no "
                "keypoints; use a -seg base (e.g. yolo11s-seg.pt)"
            )

    def resolved_settings(self, model_dir: Path) -> dict:
        """Ultralytics' own record of the run — ``run/args.yaml``, every setting after its
        defaults were applied — with the installed Ultralytics version. Read off disk and the
        package metadata, so nothing here imports the backend."""
        args_path = Path(model_dir) / "run" / "args.yaml"
        if not args_path.is_file():
            return {}
        args = yaml.safe_load(atomic.read_text(args_path)) or {}
        return {
            "backend": "ultralytics",
            "version": _installed_version("ultralytics"),
            "args": {
                key: value for key, value in args.items() if not _machine_path(value)
            },
        }

    def checkpoint_path(self, model_dir: Path) -> Path | None:
        # The checkpoint *file*, which `weights.last` records whether or not it can resume;
        # `resume` is where resumability is checked, against the loaded checkpoint.
        last = Path(model_dir) / "run" / "weights" / "last.pt"
        return last if last.is_file() else None

    def best_weights_path(self, model_dir: Path) -> Path | None:
        best = Path(model_dir) / "run" / "weights" / "best.pt"
        return best if best.is_file() else None

    def final_metrics(self, model_dir: Path) -> dict:
        """Metrics from Ultralytics' ``run/results.csv``: the **final** epoch's mAP50 (box/mask)
        *and* the **best** epoch's — the epoch ``best.pt`` was saved from (max box mAP50), so an
        interrupted or resumed run isn't judged only by its worse final epoch. ``{}`` if the file
        is missing or empty."""
        rows = self._results_rows(model_dir)
        if not rows:
            return {}

        def _num(value) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None  # a blank or malformed cell must not fail metric extraction

        def _pick(row: dict, prefix: str) -> dict:
            picked: dict[str, float] = {}
            epoch = _num(row.get("epoch"))
            if epoch is not None:
                picked[f"{prefix}epoch"] = int(epoch)
            for column, key in (
                ("metrics/mAP50(B)", "box_map50"),
                ("metrics/mAP50(M)", "mask_map50"),
                (
                    "metrics/mAP50(P)",
                    "pose_map50",
                ),  # a pose run's own (§12 *Distill keypoints*)
            ):
                value = _num(row.get(column))
                if value is not None:
                    picked[f"{prefix}{key}" if prefix == "best_" else key] = value
            return picked

        # A pose run is ranked by its keypoint mAP, a detection run by its box mAP: the
        # metric that is the point of the training is what `best.pt` is chosen by.
        rank_column = (
            "metrics/mAP50(P)" if "metrics/mAP50(P)" in rows[-1] else "metrics/mAP50(B)"
        )

        def _rank(row: dict) -> float:
            value = _num(row.get(rank_column))
            return value if value is not None else float("-inf")

        out = _pick(rows[-1], "final_")
        best = max(rows, key=_rank)
        out.update(_pick(best, "best_"))
        return out

    @staticmethod
    def _results_rows(model_dir: Path) -> list[dict]:
        """Ultralytics' ``run/results.csv``, one dict per finished epoch, or ``[]``.

        Shared by :meth:`final_metrics` and :meth:`metric_history` because they are two readings
        of one file — the summary and the series — and two copies of the read would be two places
        to notice that a column moved.

        Read fresh every call rather than cached: this file is being appended to while a training
        run is live, which is the case the curve exists for.
        """
        results = Path(model_dir) / "run" / "results.csv"
        if not results.is_file():
            return []
        with open(results, encoding="utf-8-sig") as f:
            return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]

    def metric_history(self, model_dir: Path) -> list[dict[str, float]]:
        """Per-epoch mAP50 from the same file :meth:`final_metrics` summarises.

        A row is emitted only if its epoch parses, so the partial line Ultralytics may have
        half-written when this is read mid-run is skipped rather than plotted at epoch 0 — the
        torn-tail case `AppendLog` handles for our own logs, which we do not own here.
        """

        def _num(value) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        history = []
        for row in self._results_rows(model_dir):
            epoch = _num(row.get("epoch"))
            if epoch is None:
                continue
            point: dict[str, float] = {"epoch": int(epoch)}
            for column, key in (
                ("metrics/mAP50(B)", "box_map50"),
                ("metrics/mAP50(M)", "mask_map50"),
            ):
                value = _num(row.get(column))
                if value is not None:
                    point[key] = value
            history.append(point)
        return history

    def _wire_callbacks(self, model, total_epochs, progress_cb, stop_event) -> None:
        if progress_cb is None and stop_event is None:
            return
        start = time.monotonic()
        # The mark this process began from. On a resume `epoch` starts mid-run while the
        # clock starts at zero, so dividing the run-wide fraction by it infers a throughput
        # this process never achieved — the epoch twin of ProcessingProgress's
        # recordings_at_start (DESIGN_PLAN §6: "the ETA's denominator is the work the
        # emitting coordinator did — not always the bar's"). Zero for a fresh run, which is
        # the identity, so nothing else changes.
        epochs_at_start: list[int] = []

        def _on_epoch_end(trainer) -> None:
            if stop_event is not None and stop_event.is_set():
                trainer.stop = True  # Ultralytics stops at the next epoch boundary
            if progress_cb is not None:
                metrics = {
                    k: float(v)
                    for k, v in (getattr(trainer, "metrics", None) or {}).items()
                }
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                if not epochs_at_start:
                    epochs_at_start.append(epoch - 1)
                elapsed = time.monotonic() - start
                done_here = epoch - epochs_at_start[0]
                total_here = total_epochs - epochs_at_start[0] if total_epochs else 0
                progress_cb(
                    TrainingProgressEvent(
                        epoch=epoch,
                        total_epochs=total_epochs,
                        metrics=metrics,
                        elapsed_s=elapsed,
                        # The bar spans the whole run (epoch / total_epochs); the ETA
                        # divides only by the epochs this process actually ran.
                        eta_s=(
                            remaining_seconds(done_here / total_here, elapsed)
                            if total_here > 0
                            else None
                        ),
                    )
                )

        # `on_fit_epoch_end`, not `on_train_epoch_end`: the latter fires *before* the
        # epoch is validated and before its checkpoint is written, so `trainer.metrics`
        # still holds the previous epoch's numbers — the run reported "epoch 29" carrying
        # epoch 28's mAP, and called an epoch done whose last.pt did not exist yet (#279).
        # Ultralytics' loop is on_train_epoch_end -> validate() -> save_model() ->
        # on_fit_epoch_end -> `if self.stop: break`, so the cooperative stop below still
        # takes effect at the same epoch boundary.
        model.add_callback("on_fit_epoch_end", _on_epoch_end)
