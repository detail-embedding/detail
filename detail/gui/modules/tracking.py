"""Tracking module UI (§13): config form (+ result view).

The detection-run picker offers detection **specs** (§6.5) — a run's full-lineage
kind — found across the selected recordings, annotated with ``N/M`` coverage; ``config()``
emits the chosen ``detection_spec`` and the launch resolves it to each recording's own run.
(The *arena* picker is still common-to-every-recording — arenas are user-named, not runs.)
The hyperparameter fields are built from the selected tracker's ``HYPERPARAMS_SCHEMA`` and
rebuilt when the tracker changes, so each tracker's own knobs are editable and ``config()``
only emits keys it accepts.
"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QWidget,
)

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.tracker_params import tracker_param, tracker_param_label
from detail.configs.tracking import DEFAULT_TRACKER, TrackingConfig
from detail.core.registries import TRACKERS
from detail.executor.runners import import_runner_modules
from detail.gui.forms import ConfigForm
from detail.gui.modules._common import (
    arena_covered_ids,
    arena_items,
    arena_name_groups,
    covered_ids,
    detected_class_groups,
    repopulate_combo,
    repopulate_spec_combo,
    spec_groups,
    spec_items,
    widest_group,
)
from detail.gui.help_ui import field_label
from detail.gui.modules._every_n import EveryNRow, source_rate
from detail.gui._pydantic_form import field_meta
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_module_ui,
    surfaced,
)
from detail.gui.spin import FRACTION, FloatSpin, IntSpin
from detail.gui.status_color import WARN
from detail.storage import read_cache

_NO_ARENA = "(whole frame)"
_NO_CLASS = "(every class)"
# The tracker's own knobs are a group inside Advanced Settings, so the form opens on the
# four choices that decide the run and the tuning sits one click away.
_KNOBS = "Tracker parameters"


# The tolerance knobs count frames the run *processed*, so the same number is a
# different duration at every cadence (§4). The form spells that duration out rather
# than leaving the user to divide.
_MISSED_KNOBS = ("max_missed", "max_missed_frames")


def _is_conf_knob(name: str) -> bool:
    """Whether a hyperparam is a confidence floor compared against the detection run's
    ``conf_threshold`` — ``conf_threshold`` and ``secondary_conf`` today, plus any future
    ``*_conf`` knob. IoU / merge knobs are excluded."""
    return "conf" in name


def _make_hp_widget(name: str, typ: type, default) -> QWidget:
    """A spin/check widget for one ``(type, default)`` schema entry.

    Bounds and step come from ``TRACKER_HYPERPARAM_DOCS`` — the same definition the doc
    page and the field ``?`` read — not from the default's own magnitude, which used to
    cap ``iou_weight`` at the 1.0 it defaults to.
    """
    if typ is bool:
        widget = QCheckBox()
        widget.setChecked(bool(default))
        return widget
    param = tracker_param(name)
    if typ is int:
        widget = IntSpin()
        widget.setRange(int(param.minimum), int(param.maximum))
        widget.setSingleStep(int(param.step or 1))
        widget.setValue(int(default))
        return widget
    if typ is float:
        widget = FloatSpin(FRACTION)
        widget.setRange(float(param.minimum), float(param.maximum))
        if param.step is not None:
            widget.setSingleStep(param.step)
        widget.setValue(float(default))
        return widget
    raise TypeError(f"unsupported hyperparam type {typ.__name__}")


class _TrackingConfig(ConfigWidget):
    def __init__(self):
        super().__init__()
        self._model_names: dict = {}  # {model_id: name}; filled by on_experiment_set
        import_runner_modules(PIPELINE_RUNNERS)  # populate the TRACKERS registry
        self._config_form = ConfigForm(self, TrackingConfig)
        self._form = self._config_form.main
        self._detection = QComboBox()
        self._detection.setToolTip(
            field_meta(TrackingConfig, "detection_run_id")["help"]
        )
        self._form.addRow(
            field_label("tracking", "detection_run_id", "Detection run:"),
            self._detection,
        )
        self._tracker = QComboBox()
        self._tracker.setToolTip(field_meta(TrackingConfig, "tracker_id")["help"])
        # §25's filter belongs here — the one place the form *enumerates* what a user may
        # pick — and never at the TRACKERS lookup that renders a completed run.
        self._tracker.addItems(surfaced(TRACKERS) or ["(no tracker registered)"])
        # The form opens on `configs/tracking.DEFAULT_TRACKER`, not on whichever id sorts
        # first. Those happen to be the same tracker today, which is exactly why the accident
        # went unnoticed: registering one that sorts earlier would have moved the default with
        # no edit to this file. Guarded, because `setCurrentText` on a non-editable combo is a
        # silent no-op for an id that is not there, which would restore the accident.
        default_index = self._tracker.findText(DEFAULT_TRACKER)
        if default_index >= 0:
            self._tracker.setCurrentIndex(default_index)
        self._form.addRow(
            field_label("tracking", "tracker_id", "Tracker:"), self._tracker
        )
        # Says what was substituted when a previous run's tracker is one this build does not
        # offer. On `_form`, not in the `_KNOBS` section: `advanced().clear(_KNOBS)` wipes
        # that section on every tracker change, which would take the note with it.
        self._tracker_note = self._make_conf_warning()
        self._form.addRow("", self._tracker_note)
        self._arena = QComboBox()
        self._arena.setToolTip(field_meta(TrackingConfig, "arena_ids")["help"])
        self._arena.addItem(_NO_ARENA, None)
        self._form.addRow(
            field_label("tracking", "arena_ids", "Field of view:"), self._arena
        )
        self._class = QComboBox()
        self._class.setToolTip(field_meta(TrackingConfig, "class_name")["help"])
        self._class.addItem(_NO_CLASS, None)
        self._form.addRow(field_label("tracking", "class_name", "Class:"), self._class)
        every_n_meta = field_meta(TrackingConfig, "every_n_frames")
        self._every_n = EveryNRow(TrackingConfig, upstream_noun="detected")
        self._every_n.changed.connect(self._update_missed_notes)
        self._form.addRow(
            field_label("tracking", "every_n_frames", every_n_meta["label"] + ":"),
            self._every_n,
        )
        self._recordings: list = []
        self._groups: list[dict] = []
        self._arena_groups: list[dict] = []
        self._class_groups: list[dict] = []
        self._class_resolve_gen = 0
        self._class_call = None
        # Resolved once per selection by `resolve_recordings`, off the UI thread, and read by
        # `_update_missed_notes` — which used to call `source_rate` itself, a round trip per
        # recording, on every change to the every-N spin as well as on every selection change.
        self._rates: tuple[float, float] | None = None
        self._hp_widgets: dict[str, QWidget] = {}
        self._conf_warnings: dict[str, QLabel] = {}
        self._missed_notes: dict[str, QLabel] = {}
        self._detection.currentIndexChanged.connect(self._update_conf_warnings)
        self._detection.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._arena.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._class.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._detection.currentIndexChanged.connect(self._sync_upstream_every_n)
        self._detection.currentIndexChanged.connect(self._refresh_class_groups)
        self._tracker.currentTextChanged.connect(self._rebuild_hyperparams)
        self._rebuild_hyperparams()

    def _refresh_class_groups(self, *_args) -> None:
        """Off the UI thread: which classes the *selected* detection spec's own run
        actually detected, per recording (§13). Unlike the arena list — a property of
        the recording itself — this depends on which detection run is picked, so it
        has to be re-resolved (reading each covered recording's detection zarr)
        whenever that selection changes, not just whenever the recording selection
        does. Generation-guarded like ``_load_continuity``'s pattern: a fast second
        detection-run change must not let an earlier read overwrite the newer one.
        """
        from detail.gui.workers import BackgroundCall

        spec = self._detection.currentData()
        self._class_resolve_gen += 1
        generation = self._class_resolve_gen
        recordings = list(self._recordings)
        call = BackgroundCall(
            lambda: detected_class_groups(
                recordings, spec if isinstance(spec, dict) else None
            ),
            receiver=self,
        )
        call.done.connect(
            lambda groups, g=generation: self._apply_class_groups(g, groups)
        )
        call.failed.connect(lambda _exc: None)  # the class filter is optional
        self._class_call = call
        call.start()

    def _apply_class_groups(self, generation: int, groups: list[dict]) -> None:
        if generation != self._class_resolve_gen:
            return  # a later detection-run selection has already superseded this
        self._class_groups = groups
        repopulate_combo(
            self._class,
            [(_NO_CLASS, None)]
            + arena_items(self._class_groups, len(self._recordings)),
        )

    def _sync_upstream_every_n(self, *_args) -> None:
        """Adopt the picked detection run's cadence: it is the floor this stage can use."""
        spec = self._detection.currentData()
        identity = spec.get("identity", {}) if isinstance(spec, dict) else {}
        self._every_n.set_upstream(identity.get("every_n_frames"))
        self._update_conf_warnings()
        self._update_missed_notes()

    def _selected_schema(self) -> dict:
        tracker_cls = TRACKERS.get(self._tracker.currentText())
        return (
            dict(getattr(tracker_cls, "HYPERPARAMS_SCHEMA", {})) if tracker_cls else {}
        )

    def _make_conf_warning(self) -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        palette = label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, WARN)
        label.setPalette(palette)
        label.setVisible(False)
        return label

    def _rebuild_hyperparams(self, *_args) -> None:
        # The knobs have their own layout inside Advanced Settings, so clearing them can
        # never reach one of the fixed rows above.
        self._config_form.advanced().clear(_KNOBS)
        form = self._config_form.section(_KNOBS)
        self._hp_widgets = {}
        self._conf_warnings = {}
        self._missed_notes = {}
        for name, (typ, default) in self._selected_schema().items():
            widget = _make_hp_widget(name, typ, default)
            self._hp_widgets[name] = widget
            form.addRow(
                field_label("tracking", name, f"{tracker_param_label(name)}:"), widget
            )
            if _is_conf_knob(name):
                widget.valueChanged.connect(self._update_conf_warnings)
                warning = self._make_conf_warning()
                self._conf_warnings[name] = warning
                form.addRow("", warning)
            if name in _MISSED_KNOBS:
                note = QLabel("")
                self._missed_notes[name] = note
                widget.valueChanged.connect(self._update_missed_notes)
                form.addRow("", note)
        self._update_conf_warnings()
        self._update_missed_notes()
        self._config_form.finish()

    def _update_missed_notes(self, *_args) -> None:
        """``4.0 s at this rate`` beside a frames-counted tolerance.

        The knob counts *processed* frames, so 30 is one second at every frame and four
        at every 4th. Without this the same number silently means something different on
        every run — the bug class the seconds convention exists to stop.
        """
        rates = self._rates
        every_n = self._every_n.value()
        for name, note in self._missed_notes.items():
            widget = self._hp_widgets.get(name)
            if widget is None or rates is None:
                note.setText("")
                continue
            low, high = rates
            frames = float(widget.value())
            span = f"{frames * every_n / low:.1f}"
            if low != high:
                span += f"–{frames * every_n / high:.1f}"
            note.setText(f"{span} s at this rate")

    def on_experiment_set(self, experiment) -> None:
        # A run record stores a model *id*; a picker has to show the name (§11). O(models),
        # not O(recordings), so it stays inside the UI-thread rule.
        self._model_names = experiment.model_names() if experiment is not None else {}

    def resolve_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md): the checked-selection-scoped read — this
        batch's own cadence math. (The detection-spec and field-of-view options are
        resolved separately, over every *eligible* recording, by
        ``resolve_eligible_recordings`` below — see its docstring for why those must
        not depend on which recordings happen to be checked.)
        """
        with read_cache.pinned():
            rates = source_rate(recordings)
        return {"recordings": recordings, "rates": rates}

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        """UI thread: install the resolved payload and repopulate. No disk."""
        self._recordings = resolved["recordings"]
        self._rates = resolved["rates"]
        self._every_n.set_rates(self._rates)
        self._sync_upstream_every_n()
        self._update_missed_notes()
        self._refresh_class_groups()

    def on_recordings_changed(self, recordings: list) -> None:
        """Synchronous path for direct callers (the launch dialog uses resolve→apply)."""
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def resolve_eligible_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md): every recording eligible for detection (§6.5),
        not just the checked ones ``resolve_recordings`` gets.

        Before this split, a selection made a dozen `recording.yaml` round trips per
        recording — `spec_groups` walks all four processing modules and
        `arena_name_groups` reads each recording's arenas. All of it synchronous, on
        every checkbox click, which is also what made the bug this split now fixes
        invisible for so long: checking recordings one at a time locked the detection and
        field-of-view pickers to whichever recording happened to be checked first — their
        own runs were the only ones `spec_groups`/`arena_name_groups` would ever see, so
        every other recording's showed as "not in the selected run" even when it actually
        matched. Resolved over the whole eligible pool instead, once per module pick
        rather than once per checkbox click, so it does not need to repeat the per-click
        cost.

        `read_cache.pinned()` is the other half: the record cache is mtime-keyed, so a hit
        still costs a `stat` *per property*, and this form asks for several of each recording.
        """
        with read_cache.pinned():
            groups = spec_groups(recordings, "detection", self._model_names)
            arena_groups = arena_name_groups(recordings)
        return {
            "groups": groups,
            "arena_groups": arena_groups,
            "total": len(recordings),
        }

    def apply_eligible_recordings(self, recordings: list, resolved: dict) -> None:
        """UI thread: install the resolved payload and repopulate. No disk."""
        self._groups = resolved["groups"]
        self._arena_groups = resolved["arena_groups"]
        total = resolved["total"]
        repopulate_spec_combo(
            self._detection,
            spec_items(self._groups, total)
            or [("(no detection run in this experiment)", None)],
            fallback=widest_group(self._groups)["spec"] if self._groups else None,
        )
        repopulate_combo(
            self._arena,
            [(_NO_ARENA, None)] + arena_items(self._arena_groups, total),
        )

    def covered_recording_ids(self) -> set[str] | None:
        """Recordings that have the chosen detection run, the chosen field of view, and
        the chosen class, all three. Three independent narrowings, so they intersect —
        a recording missing any one of them would otherwise be launched and validated
        against a choice it doesn't have (whole-frame instead of the named field of
        view, or ``_resolve_class_name``'s hard failure for a class the run never
        detected on that recording).
        """
        detection = covered_ids(self._groups, self._detection.currentData())
        arena = arena_covered_ids(self._arena_groups, self._arena.currentData())
        cls = arena_covered_ids(self._class_groups, self._class.currentData())
        narrowed = None
        for ids in (detection, arena, cls):
            if ids is None:
                continue
            narrowed = ids if narrowed is None else narrowed & ids
        return narrowed

    def _detection_conf(self) -> float | None:
        """The selected detection spec's confidence threshold, or None. Read straight
        from the spec identity — the same value in every recording it resolves to."""
        spec = self._detection.currentData()
        if not isinstance(spec, dict):
            return None
        conf = spec.get("identity", {}).get("conf_threshold")
        return float(conf) if isinstance(conf, (int, float)) else None

    def _update_conf_warnings(self, *_args) -> None:
        detection_conf = self._detection_conf()
        for name, warning in self._conf_warnings.items():
            widget = self._hp_widgets.get(name)
            below = (
                widget is not None
                and detection_conf is not None
                and widget.value() < detection_conf
            )
            if below:
                warning.setText(
                    f"⚠ Below the detection run's confidence ({detection_conf:.2f}). "
                    f"That run already dropped detections under {detection_conf:.2f}, so "
                    "lowering here includes no more — re-run detection at a lower "
                    "confidence for that."
                )
            warning.setVisible(below)

    def _hyperparams(self) -> dict:
        """Every knob the tracker declares, any without a widget at its schema default.

        The whole schema has to appear, not just what the form rendered: ``hyperparams`` is
        a run **identity** field, so a config that omits a knob and one that states its
        default would compare as different specs and stop matching in a batch.
        """
        result = {}
        for name, (typ, default) in self._selected_schema().items():
            widget = self._hp_widgets.get(name)
            if widget is None:
                result[name] = default
                continue
            result[name] = widget.isChecked() if typ is bool else widget.value()
        return result

    def load_config(self, config: dict) -> None:
        """Restore a previous run's tracker and its hyperparameters.

        ``detection_spec`` is a *selection* naming something this experiment mints
        sequentially, so it would bind to a different run without anything failing; the
        dialog re-asks for it.

        The field of view **is** restored now. It used to sit in the same sentence for the
        same reason — the record stored ``arena_001``, which is a different polygon on the
        next recording — and the honest response was to refuse to restore it at all. A name
        is the same thing wherever it appears, so the selection can come back.
        """
        tracker_id = config.get("tracker_id")
        substituted = bool(tracker_id) and self._tracker.findText(tracker_id) < 0
        if tracker_id and not substituted:
            self._tracker.setCurrentText(tracker_id)
            self._tracker_note.setVisible(False)
        elif substituted:
            # A run whose tracker this build does not offer (§25). Silently leaving the form
            # on its default under a row reading "seeded from this run" is the seeded-form
            # failure `tests/gui/test_start_from.py` exists for. Say which tracker it was and
            # what is selected instead.
            self._tracker_note.setText(
                f"That run used the {tracker_id!r} tracker, which this build does not "
                f"offer; {self._tracker.currentText()!r} is selected instead, with its "
                "own settings."
            )
            self._tracker_note.setVisible(True)
        arena_name = config.get("arena_name")
        index = self._arena.findData(arena_name) if arena_name else -1
        self._arena.setCurrentIndex(index if index >= 0 else 0)
        class_name = config.get("class_name")
        class_index = self._class.findData(class_name) if class_name else -1
        self._class.setCurrentIndex(class_index if class_index >= 0 else 0)
        # A substituted tracker keeps **its own** defaults. Copying the run's knobs across
        # would carry only the names the two happen to share, and a shared name is not a
        # shared meaning: `iou_threshold` was frame-to-frame association on the removed
        # `iou_tracker` and is fragment overlap on `anchor_tracker`. That leaves a form that is part one tracker
        # and part another with nothing on screen to say so — and it would make the note
        # above ("with its own settings") a lie.
        if not substituted:
            for name, value in (config.get("hyperparams") or {}).items():
                widget = self._hp_widgets.get(name)
                if widget is None:
                    continue
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(value)
        self._every_n.set_value(config.get("every_n_frames"))

    def config(self) -> dict:
        return {
            "detection_spec": self._detection.currentData(),
            "tracker_id": self._tracker.currentText(),
            "hyperparams": self._hyperparams(),
            # A **name**. `TrackingModule.validate` turns it into this recording's own
            # arena id, which is what the job and the run record store.
            "arena_name": self._arena.currentData(),
            "class_name": self._class.currentData(),
            "every_n_frames": self._every_n.value(),
        }

    def validation_error(self) -> str | None:
        if self._detection.currentData() is None:
            return "No detection run in the selected recordings — run detection first."
        return self._every_n.warning()


@register_module_ui
class TrackingModuleUI(ModuleUI):
    module_id = "tracking"

    def config_widget(self) -> ConfigWidget:
        return _TrackingConfig()

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        from detail.gui.run_preview import TrackingRunPreview

        widget = QWidget()
        layout = QVBoxLayout(widget)
        recordings = [
            (sid, info.get("run_id"))
            for sid, info in run_record.get("per_recording", {}).items()
            if info.get("status") == "complete"
        ]
        if not recordings:
            layout.addWidget(QLabel("(no completed tracking to preview)"))
            return widget
        picker = QComboBox()
        for sid, rid in recordings:
            picker.addItem(f"{sid}  ·  {rid}", (sid, rid))
        layout.addWidget(picker)
        holder = QVBoxLayout()
        layout.addLayout(holder, 1)
        current = {"widget": None}

        def show(index: int) -> None:
            sid, rid = picker.itemData(index)
            if current["widget"] is not None:
                current["widget"].deleteLater()
            current["widget"] = TrackingRunPreview(
                experiment.get_recording(sid),
                rid,
                experiment_dir=experiment.dir,
                parent_run_id=run_record.get("id"),
            )
            holder.addWidget(current["widget"])

        picker.currentIndexChanged.connect(show)
        show(0)
        return widget

    def preview_widget(
        self, recording, run_id: str, *, experiment_dir, parent_run_id
    ) -> QWidget | None:
        from detail.gui.run_preview import TrackingRunPreview

        return TrackingRunPreview(
            recording,
            run_id,
            experiment_dir=experiment_dir,
            parent_run_id=parent_run_id,
        )
