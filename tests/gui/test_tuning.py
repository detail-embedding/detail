"""Tuned detection and tracking settings in the GUI (§12 *Tune detection and tracking
settings*): the opt-in on **Distill new model**, **Tune settings…** on a trained detector's
page, the table it shows, and **Use tuned settings** on the detection and tracking forms —
the two forms a tuning is offered on, wired in the same change (``wire-every-surface``).
"""

import json
import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import fit as fit_gui  # noqa: E402
from detail.gui import tuning as gui_tuning  # noqa: E402
from detail.gui.forms import AdvancedSection  # noqa: E402
from detail.gui.modules.detection import _DetectionConfig  # noqa: E402
from detail.gui.modules.tracking import _TrackingConfig  # noqa: E402
from detail.storage import models as models_store  # noqa: E402
from detail.storage import tunings as tunings_store  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

SETTING = {
    "detection_conf": 0.05,
    "conf_threshold": 0.25,
    "secondary_conf": 0.05,
    "max_secondary": 4,
}


def _row(conf, sec, n, jumpy, default=False):
    return {
        "detection_conf": sec if n else conf,
        "conf_threshold": conf,
        "secondary_conf": sec,
        "max_secondary": n,
        "is_default": default,
        "mean_jumpy_pct": jumpy,
        "per_recording": {
            "recording_001": {
                "n_frames": 1500,
                "tracked_pct": 100.0,
                "merged_pct": 40.0 if n else 0.0,
                "grew_pct": 10.0 if n else 0.0,
                "jumpy_pct": jumpy,
                "median_area": 2000,
                "low_conf_pct": 1.0,
                "unassigned_per_frame": 0.2,
            }
        },
    }


def _record(tuning_id="tune_001", **over):
    return {
        "id": tuning_id,
        "model_id": "model_001",
        "status": "complete",
        "created_at": "2026-11-19T10:00:00Z",
        "finished_at": "2026-11-19T10:05:00Z",
        "source": {"kind": "distill", "distill_id": "distill_001"},
        "recordings": [
            {
                "recording_id": "recording_001",
                "name": "a",
                "frame_rate": 100.0,
                "sampled_frames": 1500,
                "windows": [[0, 0, 300]] * 5,
                "multi_subject_share": 0.0,
            }
        ],
        "table": [
            _row(0.5, 0.5, 2, 1.4, default=True),
            _row(0.25, 0.05, 4, 0.2),
            _row(0.7, 0.7, 0, 7.0),
        ],
        "pick": {
            "setting": SETTING,
            "keep_defaults": False,
            "reason": "lowest mean jumpy share (0.20%)",
        },
        "warnings": ["a warning"],
        **over,
    }


def _experiment(tmp_path, *, tuned=True):
    experiment = Experiment.create("p", tmp_path / "exp")
    directory = models_store.model_dir(experiment.dir, "model_001")
    directory.mkdir(parents=True)
    models_store.write_model_yaml(
        directory,
        {
            "id": "model_001",
            "name": "student",
            "status": "complete",
            "task": "segment",
            "trainer_id": "ultralytics_yolo",
            "classes": [{"class_id": 0, "name": "subject"}],
        },
    )
    if tuned:
        tdir = tunings_store.tuning_dir(experiment.dir, "model_001", "tune_001")
        tunings_store.write_tuning(tdir, _record())
        (tdir / "job.json").write_text(
            json.dumps({"schema_version": 1, "status": "complete"}), encoding="utf-8"
        )
    return experiment


def _in_advanced(widget) -> bool:
    node = widget.parent()
    while node is not None and not isinstance(node, AdvancedSection):
        node = node.parent()
    return node is not None


# ------------------------------------------------------------------ distill


def test_the_distill_opt_in_is_off_and_under_advanced(qapp, tmp_path):
    dialog = fit_gui.DistillDialog(_experiment(tmp_path, tuned=False))
    assert not dialog._tune.isChecked()
    assert _in_advanced(dialog._tune) and _in_advanced(dialog._tune_seconds)
    assert "adds detection time" in dialog._tune.toolTip()
    assert not dialog._tune_seconds.isEnabled()
    dialog._tune.setChecked(True)
    assert dialog._tune_seconds.isEnabled()


def test_the_distill_passes_the_choice(qapp, tmp_path, monkeypatch):
    experiment = _experiment(tmp_path, tuned=False)
    experiment.add_recording("a")
    dialog = fit_gui.DistillDialog(experiment)
    seen = {}
    monkeypatch.setattr(
        experiment, "distill", lambda name, **kw: seen.update(kw) or object()
    )
    dialog._name.setText("m")
    dialog._class_rows[0][0].setText("subject")
    dialog._class_rows[0][1].setText("a subject")
    dialog._tune.setChecked(True)
    dialog._tune_seconds.setValue(120)
    dialog._run()
    assert seen["tune"] is True
    assert seen["tune_sample_seconds"] == 120.0


# ------------------------------------------------------------------ detection form


def _select_model(form, model_id):
    for i in range(form._model.count()):
        data = form._model.itemData(i)
        if isinstance(data, dict) and data["id"] == model_id:
            form._model.setCurrentIndex(i)
            return
    raise AssertionError(f"{model_id} is not offered")


def test_detection_offers_the_tuned_confidence(qapp, tmp_path):
    form = _DetectionConfig()
    form.on_experiment_set(_experiment(tmp_path))
    _select_model(form, "model_001")
    assert form._tuned.button.isEnabled()
    assert "tune_001 (from distill distill_001" in form._tuned.note.text()
    form._conf.setValue(0.5)
    form._tuned.button.click()
    assert form._conf.value() == pytest.approx(0.05)
    assert form.config()["conf_threshold"] == pytest.approx(0.05)


def test_detection_says_why_an_untuned_model_offers_nothing(qapp, tmp_path):
    form = _DetectionConfig()
    form.on_experiment_set(_experiment(tmp_path, tuned=False))
    _select_model(form, "model_001")
    assert not form._tuned.button.isEnabled()
    assert "Tune settings…" in form._tuned.button.toolTip()
    _select_model(form, "sam3")
    assert "trained in this experiment" in form._tuned.button.toolTip()


# ------------------------------------------------------------------ tracking form


def _pick_detection(form, conf, weights_source="trained"):
    identity = {
        "model_id": "model_001",
        "weights_source": weights_source,
        "conf_threshold": conf,
        "every_n_frames": 1,
    }
    form._detection.addItem("det", {"identity": identity})
    form._detection.setCurrentIndex(form._detection.count() - 1)


def test_tracking_fills_the_three_knobs(qapp, tmp_path):
    form = _TrackingConfig()
    form.on_experiment_set(_experiment(tmp_path))
    _pick_detection(form, 0.05)
    assert form._tuned.button.isEnabled()
    form._tuned.button.click()
    hp = form.config()["hyperparams"]
    assert (hp["conf_threshold"], hp["secondary_conf"], hp["max_secondary"]) == (
        0.25,
        0.05,
        4,
    )


def test_tracking_warns_when_the_detection_kept_too_little(qapp, tmp_path):
    form = _TrackingConfig()
    form.on_experiment_set(_experiment(tmp_path))
    _pick_detection(form, 0.2)
    assert form._tuned.button.isEnabled()
    assert "Run detection again at that confidence" in form._tuned.warning.text()


def test_tracking_offers_nothing_for_another_tracker(qapp, tmp_path):
    form = _TrackingConfig()
    form.on_experiment_set(_experiment(tmp_path))
    _pick_detection(form, 0.05)
    other = [form._tracker.itemText(i) for i in range(form._tracker.count())]
    other = [t for t in other if t != "anchor_tracker"]
    if not other:
        pytest.skip("only one tracker registered")
    form._tracker.setCurrentText(other[0])
    assert not form._tuned.button.isEnabled()
    assert "anchor_tracker" in form._tuned.button.toolTip()


# ------------------------------------------------------------------ the table


def test_the_table_shows_every_setting_with_the_pick_and_defaults_marked(qapp):
    view = gui_tuning.TuningView(_record())
    assert view.model.rowCount() == 3
    marks = [view.model.item(r, 8).text() for r in range(3)]
    assert marks == ["picked", "defaults", ""]  # steadiest first
    assert "Recommended: detection confidence 0.05" in view.summary.text()


def test_keep_defaults_is_said_as_such(qapp):
    record = _record(
        pick={"setting": SETTING, "keep_defaults": True, "reason": "as stable"}
    )
    assert gui_tuning.TuningView(record).summary.text().startswith("Keep the defaults")


# ------------------------------------------------------------------ Tune settings…


class _FakeTune:
    def __init__(self, record):
        self.id = record["id"]
        self.status = "complete"
        self.error = None
        self._record = record

    def progress(self):
        from detail.pipeline.tune_tracking import TuneProgressEvent

        yield TuneProgressEvent("detecting", 5, 10, 1.0, 1.0, "detecting a")
        yield TuneProgressEvent("sweeping", 10, 43, 2.0, 2.0, "setting 10/43")

    def wait(self):
        return self

    def cancel(self):
        pass

    def record(self):
        return self._record


def test_tune_settings_runs_off_thread_and_shows_the_table(qapp, tmp_path, monkeypatch):
    experiment = _experiment(tmp_path, tuned=False)
    calls = []

    def _tune(model_id, ids, **kw):
        calls.append((model_id, ids, kw))
        return _FakeTune(_record())

    monkeypatch.setattr(experiment, "tune_tracking", _tune)
    dialog = gui_tuning.TuneDialog(experiment, "model_001")
    dialog._seconds.setValue(60)
    dialog.run()
    assert not dialog._tune.isEnabled()
    drain_until(qapp, lambda: dialog.record is not None)
    ((model_id, ids, kw),) = calls
    assert model_id == "model_001" and ids is None
    assert kw["sample_seconds"] == 60.0
    assert dialog._tune.isEnabled()
    assert "Use tuned settings" in dialog._status.text()


def test_the_model_page_offers_tune_and_shows_the_newest(qapp, tmp_path):
    from detail.gui.pages import ModelDetailPage

    page = ModelDetailPage(_experiment(tmp_path), "model_001")
    texts = [b.text() for b in page.findChildren(QPushButton)]
    assert "Tune settings…" in texts and "Show tuning…" in texts


def test_an_untuned_model_page_offers_tune_only(qapp, tmp_path):
    from detail.gui.pages import ModelDetailPage

    page = ModelDetailPage(_experiment(tmp_path, tuned=False), "model_001")
    texts = [b.text() for b in page.findChildren(QPushButton)]
    assert "Tune settings…" in texts and "Show tuning…" not in texts


def test_a_partial_distill_offers_resume_and_shows_its_tune_stage(qapp, tmp_path):
    from detail.gui.pages import DistillRunDetail
    from detail.storage import distill_runs as distill_store

    experiment = _experiment(tmp_path)
    distill_store.write_run(
        experiment.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "m",
            "status": "partial",
            "created_at": "2026-11-19",
            "config": {"tune": {"sample_seconds": 300.0}},
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "complete", "model_id": "model_001"},
                "tune": {"status": "failed", "model_id": "model_001"},
            },
            "error": "tuning failed: out of memory",
        },
    )
    page = DistillRunDetail(experiment, "distill_001")
    texts = [b.text() for b in page.findChildren(QPushButton)]
    assert "Resume distill" in texts
    assert "Show tuning…" not in texts


def test_the_row_starts_unavailable_with_a_reason(qapp):
    row = gui_tuning.TunedSettingsRow(lambda s: None)
    assert not row.button.isEnabled()
    assert row.button.toolTip()
    row.offer(types.SimpleNamespace(setting=None, reason="nothing tracked"))
    assert row.note.text() == "nothing tracked"
