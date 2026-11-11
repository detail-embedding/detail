"""UltralyticsYOLOTrainer: registration, hyperparams, readiness (no live train)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from detail.configs.training import UltralyticsYOLOConfig
from detail.core.registries import TRAINERS
from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer


def test_registered_and_attrs():
    assert TRAINERS.get("ultralytics_yolo") is UltralyticsYOLOTrainer
    t = UltralyticsYOLOTrainer()
    assert t.model_family == "yolo"
    assert t.supported_backends == ["cpu", "cuda", "mps"]
    # closed-set: the trained model takes no prompt cues (§12 design decision)
    assert t.supported_cue_types == []
    # hyperparams are sourced from the pydantic config model (§12 migration)
    assert t.config_model is UltralyticsYOLOConfig


def test_validate_hyperparams_fills_defaults():
    out = UltralyticsYOLOTrainer().validate_hyperparams({"epochs": 50})
    assert out == {"epochs": 50, "batch_size": 32, "imgsz": 640, "patience": 100}


def test_validate_hyperparams_rejects_unknown_key():
    # extra="forbid" on the config model — a pydantic ValidationError (a ValueError)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        UltralyticsYOLOTrainer().validate_hyperparams({"learning_rate": 0.01})


def test_validate_hyperparams_type_check():
    with pytest.raises(ValueError, match="valid integer"):
        UltralyticsYOLOTrainer().validate_hyperparams({"epochs": "lots"})


def test_validate_hyperparams_patience_bounds():
    t = UltralyticsYOLOTrainer()
    assert t.validate_hyperparams({"patience": 0})["patience"] == 0  # 0 disables
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        t.validate_hyperparams({"patience": -1})


def test_validate_hyperparams_epochs_lower_bound():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        UltralyticsYOLOTrainer().validate_hyperparams({"epochs": 0})


def test_train_passes_patience_to_model_train(tmp_path, monkeypatch):
    """``patience`` (and the other hyperparams) reach Ultralytics' ``model.train``.

    A fake ``ultralytics`` module records the kwargs the trainer passes, so the
    early-stopping knob is verified without a live GPU train. Scoped via
    ``monkeypatch.setitem`` — auto-reverted, never a module-level ``sys.modules`` mock.
    """
    from detail.core.dataclasses import ComputeTarget, TrainingJob
    from detail.storage import datasets as datasets_store
    from detail.trainers import ultralytics_yolo

    captured: dict = {}

    class _FakeYOLO:
        def __init__(self, weights):
            self._weights = weights

        def add_callback(self, *_args, **_kwargs):
            pass

        def train(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setattr(ultralytics_yolo, "_weights_dir", lambda: tmp_path / "cache")

    dataset_dir = tmp_path / "dataset_001"
    dataset_dir.mkdir()
    datasets_store.write_data_yaml(dataset_dir, {0: "mouse"})
    model_dir = tmp_path / "models" / "model_001"
    model_dir.mkdir(parents=True)

    def _job(hyperparams):
        return TrainingJob(
            job_id="model_001",
            model_id="model_001",
            dataset_id="dataset_001",
            trainer_id="ultralytics_yolo",
            base_model="yolo11s-seg.pt",
            hyperparams=hyperparams,
            compute=ComputeTarget("cpu", None, None, None),
            created_at="2026-07-10T00:00:00Z",
        )

    trainer = UltralyticsYOLOTrainer()
    trainer.train(_job({"patience": 7}), dataset_dir, model_dir, "cpu")
    assert captured["patience"] == 7
    assert captured["epochs"] == 100  # default from the model

    captured.clear()
    trainer.train(_job({}), dataset_dir, model_dir, "cpu")
    assert captured["patience"] == 100  # default when unset

    # Ultralytics' own argument names, which are *not* ours to rename. `project` + `name`
    # are where it writes: model_dir/run/, exactly where checkpoint_path and
    # best_weights_path look for weights afterwards.
    assert captured["project"] == str(model_dir)
    assert captured["name"] == "run"


def test_train_uses_ultralytics_own_argument_names(tmp_path, monkeypatch):
    """A third-party keyword is not ours to rename, and renaming one broke every train.

    The Project -> Experiment rename (#69) rewrote ``project=`` to ``experiment=`` here.
    Ultralytics rejects an unknown argument, so distill training crashed on launch for
    everyone, every time — and it survived for weeks because training is the one stage no
    test runs (no GPU, no dataset). This pins the names against the next global rename.
    """
    from detail.core.dataclasses import ComputeTarget, TrainingJob
    from detail.storage import datasets as datasets_store
    from detail.trainers import ultralytics_yolo

    captured: dict = {}

    class _StrictYOLO:
        """Rejects unknown kwargs the way Ultralytics does."""

        _ACCEPTED = {
            "data",
            "epochs",
            "batch",
            "imgsz",
            "patience",
            "project",
            "name",
            "device",
            "exist_ok",
            "verbose",
            "resume",
        }

        def __init__(self, weights):
            self._weights = weights

        def add_callback(self, *_args, **_kwargs):
            pass

        def train(self, **kwargs):
            unknown = set(kwargs) - self._ACCEPTED
            if unknown:
                raise SyntaxError(f"unknown train argument(s): {sorted(unknown)}")
            captured.update(kwargs)

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = _StrictYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setattr(ultralytics_yolo, "_weights_dir", lambda: tmp_path / "cache")

    dataset_dir = tmp_path / "dataset_001"
    dataset_dir.mkdir()
    datasets_store.write_data_yaml(dataset_dir, {0: "mouse"})
    model_dir = tmp_path / "models" / "model_001"
    model_dir.mkdir(parents=True)

    UltralyticsYOLOTrainer().train(
        TrainingJob(
            job_id="model_001",
            model_id="model_001",
            dataset_id="dataset_001",
            trainer_id="ultralytics_yolo",
            base_model="yolo11s-seg.pt",
            hyperparams={},
            compute=ComputeTarget("cpu", None, None, None),
            created_at="2026-07-10T00:00:00Z",
        ),
        dataset_dir,
        model_dir,
        "cpu",
    )
    # ...and the output lands where the weights accessors go looking for it.
    assert Path(captured["project"]) / captured["name"] == model_dir / "run"


def test_is_ready_reflects_ultralytics_presence(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert UltralyticsYOLOTrainer().is_ready() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert UltralyticsYOLOTrainer().is_ready() is True


def test_resolved_settings_are_args_yaml_with_version_and_no_machine_paths(tmp_path):
    """Ultralytics resolves every setting this tool does not pass from its own defaults; its
    ``run/args.yaml`` is the record of them. Copied into model.yaml with the installed version,
    minus every value that is a path on this machine (no absolute paths in experiment files).
    """
    import importlib.metadata

    import yaml

    run = tmp_path / "run"
    run.mkdir()
    (run / "args.yaml").write_text(
        yaml.safe_dump(
            {
                "task": "segment",
                "model": "yolo11s-seg.pt",
                "data": "/Users/someone/cache/data_configs/abc.yaml",
                "epochs": 100,
                "batch": 32,
                "lr0": 0.01,
                "project": "/home/someone/exp/models/model_001",
                "save_dir": "C:\\exp\\models\\model_001\\run",
                "name": "run",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    got = UltralyticsYOLOTrainer().resolved_settings(tmp_path)
    assert got["backend"] == "ultralytics"
    assert got["version"] == importlib.metadata.version("ultralytics")
    assert got["args"] == {
        "task": "segment",
        "model": "yolo11s-seg.pt",
        "epochs": 100,
        "batch": 32,
        "lr0": 0.01,
        "name": "run",
        "seed": 0,
    }
    assert UltralyticsYOLOTrainer().resolved_settings(tmp_path / "nothing") == {}


def test_weights_paths_none_before_training(tmp_path):
    t = UltralyticsYOLOTrainer()
    assert t.best_weights_path(tmp_path) is None
    assert t.checkpoint_path(tmp_path) is None
    assert t.final_metrics(tmp_path) == {}  # no results.csv yet


def test_training_data_yaml_uses_absolute_path(tmp_path, monkeypatch):
    """The trainer hands Ultralytics a data.yaml whose ``path`` is the *absolute*
    dataset dir (a copy in the weights cache), while the on-disk descriptor stays
    relative for portability. Ultralytics resolves a relative ``path:`` against the
    cwd — which the trainer chdirs into the weights cache — so the relative form
    would point at the wrong place (the bug this fixes)."""
    import yaml

    from detail.storage import datasets as datasets_store
    from detail.trainers import ultralytics_yolo

    monkeypatch.setattr(ultralytics_yolo, "_weights_dir", lambda: tmp_path / "cache")
    dataset_dir = tmp_path / "dataset_001"
    dataset_dir.mkdir()
    datasets_store.write_data_yaml(dataset_dir, {0: "mouse"})

    out = UltralyticsYOLOTrainer()._training_data_yaml(
        dataset_dir, tmp_path / "models" / "model_001"
    )
    spec = yaml.safe_load(out.read_text(encoding="utf-8-sig"))
    assert spec["path"] == str(dataset_dir.resolve())  # absolute, cwd-independent
    assert spec["train"] == "images/train" and spec["val"] == "images/val"
    assert spec["names"] == {0: "mouse"}
    # the experiment descriptor is untouched — still portable
    on_disk = yaml.safe_load(
        (dataset_dir / "data.yaml").read_text(encoding="utf-8-sig")
    )
    assert on_disk["path"] == "."


# --- progress reports the epoch its metrics were measured on (#279) ---------- #
def _fake_yolo_module(monkeypatch, tmp_path, yolo_cls):
    from detail.trainers import ultralytics_yolo

    module = types.ModuleType("ultralytics")
    module.YOLO = yolo_cls
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(ultralytics_yolo, "_weights_dir", lambda: tmp_path / "cache")


def _training_job(hyperparams):
    from detail.core.dataclasses import ComputeTarget, TrainingJob

    return TrainingJob(
        job_id="model_001",
        model_id="model_001",
        dataset_id="dataset_001",
        trainer_id="ultralytics_yolo",
        base_model="yolo11s-seg.pt",
        hyperparams=hyperparams,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-07-10T00:00:00Z",
    )


def _dataset_and_model_dirs(tmp_path):
    from detail.storage import datasets as datasets_store

    dataset_dir = tmp_path / "dataset_001"
    dataset_dir.mkdir()
    datasets_store.write_data_yaml(dataset_dir, {0: "mouse"})
    model_dir = tmp_path / "models" / "model_001"
    model_dir.mkdir(parents=True)
    return dataset_dir, model_dir


class _FakeUltralyticsTrainer:
    """Stands in for Ultralytics' BaseTrainer as the progress callbacks see it."""

    def __init__(self):
        self.epoch = 0
        self.metrics: dict = {}
        self.stop = False


def test_progress_event_carries_the_metrics_of_the_epoch_it_names(
    tmp_path, monkeypatch
):
    """The event for epoch N carries epoch N's *validated* metrics, not N-1's.

    Ultralytics' loop is ``on_train_epoch_end`` -> ``validate()`` -> ``save_model()`` ->
    ``on_fit_epoch_end`` (engine/trainer.py, 8.4.70). Wired to the first hook, the trainer
    reports epoch N as done while ``trainer.metrics`` still holds N-1's numbers and before
    N's checkpoint exists — #279 saw "epoch 29" carrying epoch 28's mAP. The fake below
    fires both hooks in that real order; rewire to ``on_train_epoch_end`` and the asserted
    pairs shift by an epoch.
    """
    measured = [{"metrics/mAP50(B)": 0.10}, {"metrics/mAP50(B)": 0.20}]

    class _FakeYOLO:
        def __init__(self, weights):
            self._callbacks: dict = {}

        def add_callback(self, name, fn):
            self._callbacks.setdefault(name, []).append(fn)

        def train(self, **_kwargs):
            trainer = _FakeUltralyticsTrainer()
            for index, epoch_metrics in enumerate(measured):
                trainer.epoch = index
                for fn in self._callbacks.get("on_train_epoch_end", []):
                    fn(trainer)  # fires BEFORE validation
                trainer.metrics = (
                    epoch_metrics  # validate() writes this epoch's numbers
                )
                for fn in self._callbacks.get("on_fit_epoch_end", []):
                    fn(trainer)  # fires AFTER validation and after save_model()

    _fake_yolo_module(monkeypatch, tmp_path, _FakeYOLO)
    dataset_dir, model_dir = _dataset_and_model_dirs(tmp_path)

    events = []
    UltralyticsYOLOTrainer().train(
        _training_job({"epochs": 2}),
        dataset_dir,
        model_dir,
        "cpu",
        progress_cb=events.append,
    )

    # events[0] is the "loading base model" spin-up notice (epoch 0, no metrics).
    reported = [(e.epoch, e.metrics.get("metrics/mAP50(B)")) for e in events if e.epoch]
    assert reported == [(1, 0.10), (2, 0.20)]


def test_a_stop_event_still_stops_at_the_epoch_boundary(tmp_path, monkeypatch):
    """Moving the callback must not cost the cooperative cancel.

    Deliberately **hook-agnostic**: the fake fires both hooks in Ultralytics' real order,
    so this passes whichever one the trainer registered on. That is the point — the
    property is "a set stop_event stops the run", which must hold either way, and a
    version that fires only ``on_fit_epoch_end`` would just be a second copy of the
    hook-name assertion above. Ultralytics checks ``if self.stop: break`` after
    ``on_fit_epoch_end`` (engine/trainer.py:577), so the later hook is still in time.
    """
    import threading

    seen: dict = {}

    class _FakeYOLO:
        def __init__(self, weights):
            self._callbacks: dict = {}

        def add_callback(self, name, fn):
            self._callbacks.setdefault(name, []).append(fn)

        def train(self, **_kwargs):
            trainer = _FakeUltralyticsTrainer()
            for hook in ("on_train_epoch_end", "on_fit_epoch_end"):
                for fn in self._callbacks.get(hook, []):
                    fn(trainer)
            seen["stop"] = trainer.stop

    _fake_yolo_module(monkeypatch, tmp_path, _FakeYOLO)
    dataset_dir, model_dir = _dataset_and_model_dirs(tmp_path)

    stop_event = threading.Event()
    stop_event.set()
    UltralyticsYOLOTrainer().train(
        _training_job({"epochs": 5}),
        dataset_dir,
        model_dir,
        "cpu",
        stop_event=stop_event,
    )
    assert seen["stop"] is True
