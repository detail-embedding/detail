"""Tuned detection and tracking settings in the GUI (§12 *Tune detection and tracking
settings*).

Three pieces, one per place a user meets a tuning:

- :class:`TuningView` shows one tuning: which it is and who made it, the pick and why, any
  warning, and the whole sweep as a sortable table (each setting's lowest tracked share, mean
  jumpy share, merged and grew shares, and the detection confidence it needs).
- :class:`TuneDialog` is **Tune settings…** on a trained detector's page: choose recordings and
  the sample size, then ``Experiment.tune_tracking`` runs off the UI thread
  (``busy.run_progress``, with Cancel) and the result is shown in place.
- :class:`TunedSettingsRow` is **Use tuned settings** on the detection and tracking forms. It
  offers the newest complete tuning of the model in play — from a distill or a standalone tune,
  saying which — and writes its values into the form's own fields when pressed. Nothing else
  defaults to a tuning.

The computation is ``detail.analysis.track_quality`` and ``detail.pipeline.tune_tracking``;
this module only shows it and launches it.
"""

from collections.abc import Callable

from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.configs.tuning import TuningConfig
from detail.gui.busy import run_busy, run_progress
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.dialog import Dialog
from detail.gui.recording_items import (
    fill_sampling_list,
    sampling_label,
    selected_recording_ids,
)
from detail.gui._pydantic_form import field_widget
from detail.gui.sizing import set_available
from detail.gui.status_color import MUTED, WARN, set_label_color
from detail.gui.tables import SortCell, fit_table_height, fit_table_width, list_table
from detail.storage import read_cache

#: What a tuning cannot see, on screen beside every table and in the docs (``docs/gui.md``).
LIMIT_NOTE = (
    "Measured without labels on a sample of each recording. A low jumpy share means the "
    "mask is steady, not that it is right: a piece of background merged on every frame is "
    "steady too, so check merged and grew beside it and look at a few frames in the tracking "
    "viewer before a long run. The tracker follows one subject; the field of view is not "
    "applied to the sample."
)

_COLUMNS = (
    "Anchor conf",
    "Secondary conf",
    "Max merged",
    "Detection conf",
    "Tracked % (lowest)",
    "Jumpy % (mean)",
    "Merged %",
    "Grew %",
    "",
)


def _num(value: float, digits: int = 2) -> SortCell:
    return SortCell(f"{value:.{digits}f}", float(value))


def _mean(per: list[dict], key: str) -> float:
    return sum(q[key] for q in per) / len(per) if per else 0.0


class TuningView(QWidget):
    """One tuning record (``tuning.yaml``), read-only."""

    def __init__(self, record: dict, parent=None) -> None:
        super().__init__(parent)
        from detail.api.tuning import source_label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        head = QLabel(
            f"{record.get('id')} of {record.get('model_id')}, from "
            f"{source_label(record)} — {record.get('status')}"
        )
        head.setWordWrap(True)
        layout.addWidget(head)
        sampled = [
            f"{r['name']}: {r['sampled_frames']} frames in {len(r['windows'])} windows"
            for r in record.get("recordings") or []
        ]
        if sampled:
            sample = QLabel("Sample: " + "; ".join(sampled))
            sample.setWordWrap(True)
            layout.addWidget(sample)
        pick = record.get("pick") or {}
        setting = pick.get("setting")
        if setting and not pick.get("keep_defaults"):
            text = (
                f"Recommended: detection confidence {setting['detection_conf']:g}; "
                f"tracker {setting['conf_threshold']:g} / {setting['secondary_conf']:g} / "
                f"max {setting['max_secondary']}. {pick.get('reason', '')}"
            )
        elif setting:
            text = f"Keep the defaults: {pick.get('reason', '')}"
        else:
            text = (
                f"No recommendation: {pick.get('reason') or record.get('error') or ''}"
            )
        self.summary = QLabel(text)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        for warning in record.get("warnings") or []:
            note = QLabel(warning)
            note.setWordWrap(True)
            set_label_color(note, WARN)
            layout.addWidget(note)
        self.model = QStandardItemModel(0, len(_COLUMNS))
        self.model.setHorizontalHeaderLabels(list(_COLUMNS))
        chosen = (
            (
                setting["conf_threshold"],
                setting["secondary_conf"],
                setting["max_secondary"],
            )
            if setting
            else None
        )
        for row in sorted(
            record.get("table") or [],
            key=lambda r: (r["mean_jumpy_pct"], r["max_secondary"]),
        ):
            per = list(row["per_recording"].values())
            key = (row["conf_threshold"], row["secondary_conf"], row["max_secondary"])
            marks = []
            if key == chosen:
                marks.append("picked")
            if row.get("is_default"):
                marks.append("defaults")
            self.model.appendRow(
                [
                    _num(row["conf_threshold"]),
                    _num(row["secondary_conf"]),
                    SortCell(str(row["max_secondary"]), row["max_secondary"]),
                    _num(row["detection_conf"]),
                    _num(min((q["tracked_pct"] for q in per), default=0.0)),
                    _num(row["mean_jumpy_pct"]),
                    _num(_mean(per, "merged_pct")),
                    _num(_mean(per, "grew_pct")),
                    SortCell(", ".join(marks), ", ".join(marks)),
                ]
            )
        self.table = list_table()
        self.table.setModel(self.model)
        self.table.resizeColumnsToContents()
        fit_table_height(self.table)
        fit_table_width(self.table)
        layout.addWidget(self.table)
        note = QLabel(LIMIT_NOTE)
        note.setWordWrap(True)
        set_label_color(note, MUTED)
        layout.addWidget(note)


class TuningDialog(Dialog):
    """A tuning shown on its own (**Show tuning…** on a model or distill page)."""

    fits_content = True

    def __init__(self, tuning: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Tuning {tuning.get('id')}")
        self.view = TuningView(tuning)
        self.body().addWidget(self.view)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        self.set_buttons(box)


class TuneDialog(Dialog):
    """**Tune settings…** for one trained detector (module docstring)."""

    fits_content = True

    def __init__(self, experiment, model_id: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Tune settings: {model_id}")
        self._experiment = experiment
        self._model_id = model_id
        self._running = False
        self.record: dict | None = None

        body = self.body()
        intro = QLabel(
            "Run the model on a sample of the recordings at a low confidence, try a grid of "
            "detection confidence and tracker merging settings, and recommend the steadiest "
            "mask that still tracks. Nothing changes until you press Use tuned settings on "
            "the detection or tracking form."
        )
        intro.setWordWrap(True)
        body.addWidget(intro)
        form = QFormLayout()
        self._recordings = QListWidget()
        self._recordings.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._recordings_label = QLabel(sampling_label(0))
        form.addRow(self._recordings_label, self._recordings)
        meta = TuningConfig.model_fields["sample_seconds"].json_schema_extra
        self._seconds = field_widget(TuningConfig, "sample_seconds")
        form.addRow(meta["label"] + ":", self._seconds)
        self._compute = ComputeTargetRow()
        self._compute.add_to(form)
        body.addLayout(form)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        body.addWidget(self._status)
        self._result = QVBoxLayout()
        body.addLayout(self._result)
        body.addStretch()

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._tune = QPushButton("Tune")
        self._tune.clicked.connect(self.run)
        box.addButton(self._tune, QDialogButtonBox.ButtonRole.ActionRole)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._sync()
        # O(recordings) on what may be a share: read off the UI thread (CLAUDE.md).
        run_busy(
            self._recordings,
            lambda: _recordings_of(experiment),
            self._fill_recordings,
            label="Reading recordings…",
        )

    def _fill_recordings(self, recordings) -> None:
        with read_cache.pinned():
            with_video = fill_sampling_list(self._recordings, recordings)
        self._recordings_label.setText(sampling_label(with_video))

    def run(self) -> None:
        """Tune off the UI thread (``Experiment.tune_tracking``), with a bar and Cancel."""
        from detail.core.progress import OperationCancelled

        experiment, model_id = self._experiment, self._model_id
        recording_ids = selected_recording_ids(self._recordings) or None
        seconds = float(self._seconds.value())
        compute = self._compute.compute_target()
        self._running = True
        self._sync()

        def work(emit):
            run = experiment.tune_tracking(  # off-thread
                model_id, recording_ids, sample_seconds=seconds, compute=compute
            )
            try:
                for event in run.progress():
                    total = event.frames_total or 0
                    fraction = event.frames_done / total if total else 0.0
                    bar = (
                        0.8 * fraction
                        if event.phase == "detecting"
                        else 0.8 + 0.2 * fraction
                    )
                    emit(event.message or event.phase, bar, event.phase)
            except OperationCancelled:
                run.cancel()
                run.wait()
                raise
            run.wait()
            if run.status != "complete":
                raise RuntimeError(run.error or f"tuning {run.status}")
            return run.record()

        run_progress(
            self,
            work,
            self.show_record,
            label="Tuning…",
            on_error=self._failed,
            on_cancel=self._cancelled,
            supersedes="tune",
        )

    def show_record(self, record: dict) -> None:
        """Show a finished tuning in the window."""
        self._running = False
        self.record = record
        while self._result.count():
            item = self._result.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._result.addWidget(TuningView(record))
        self._status.setText(
            "Done. The detection and tracking forms now offer Use tuned settings for "
            "this model."
        )
        self._sync()

    def _cancelled(self) -> None:
        self._running = False
        self._status.setText("Cancelled. Tune runs it again.")
        self._sync()

    def _failed(self, exc: BaseException) -> None:
        self._running = False
        self._status.setText(f"Tuning did not finish: {exc}")
        self._sync()

    def _sync(self) -> None:
        set_available(self._tune, not self._running, "Tuning is running")


class TunedSettingsRow(QWidget):
    """**Use tuned settings** and a line saying which tuning it would use.

    The form calls :meth:`offer` whenever its model (or detection run) changes, with the
    tuning found or the reason there is none; pressing the button calls ``on_use`` with the
    tuned :class:`~detail.analysis.track_quality.Setting`, and the form writes its fields.
    """

    def __init__(self, on_use: Callable, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.button = QPushButton("Use tuned settings")
        self.button.setToolTip(
            "Fill this form with the settings the newest tuning of this model recommends."
        )
        self.button.clicked.connect(self._use)
        row.addWidget(self.button)
        row.addStretch()
        layout.addLayout(row)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        set_label_color(self.note, MUTED)
        layout.addWidget(self.note)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        set_label_color(self.warning, WARN)
        self.warning.hide()
        layout.addWidget(self.warning)
        self._on_use = on_use
        self.found = None
        self.offer(None, "Choose a model this experiment trained and tuned")

    def offer(
        self, found, reason: str | None = None, warning: str | None = None
    ) -> None:
        """Offer ``found`` (a ``TunedSettings``), or say ``reason`` there is none;
        ``warning`` is shown apart, in amber."""
        self.found = found
        self.warning.setText(warning or "")
        self.warning.setVisible(bool(warning))
        if found is None or found.setting is None:
            why = reason or (found.reason if found is not None else "")
            set_available(self.button, False, why)
            self.note.setText(why)
            return
        set_available(self.button, True, "")
        self.note.setText(f"From {found.describe()}. {found.reason}")

    def _use(self) -> None:
        if self.found is not None and self.found.setting is not None:
            self._on_use(self.found.setting)


def _recordings_of(experiment) -> list:
    with read_cache.pinned():
        return list(experiment.recordings)  # off-thread (run_busy)


def untuned_reason(model_id: str) -> str:
    return (
        f"{model_id} has not been tuned: open it under Models and press Tune settings…, or "
        "tick Tune detection and tracking settings when distilling"
    )
