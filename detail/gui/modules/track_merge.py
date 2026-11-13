"""Track-merge module UI (§13): config form.

Two tracking-spec pickers (mirroring `_TrackingConfig`'s own detection-spec picker) let
a user combine two class-filtered single-subject tracks (one per animal) into one
multi-subject run. Fixed at two for now — every concrete use of this feature so far is
two animals — but the config model and backend module already take any N ≥ 2; adding a
third picker row later is the same widget repeated, not a redesign.

The stage merges tracks of one detection run only, so every row after the first follows
Track A: a tracking run on another detection run (recording by recording) is greyed, saying so.

Nothing here is restorable on "start from a previous run": a ``source_specs`` entry
names two tracking runs this experiment may not have under that spec any more, the same
reason `_TrackingConfig` never restores its own `detection_spec` either — the dialog
re-asks. `ConfigWidget.load_config`'s default (`config_model` left unset) is already a
no-op, so nothing needs overriding here.
"""

from PyQt6.QtWidgets import QComboBox, QLabel, QWidget

from detail.api import run_spec
from detail.configs.track_merge import TrackMergeConfig
from detail.gui._pydantic_form import field_meta
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import (
    UpstreamSpecPicker,
    lineage_fits,
    spec_groups,
    spec_reads,
)
from detail.gui.modules.registry import ConfigWidget, ModuleUI, register_module_ui
from detail.storage import read_cache

_ROW_LABELS = ("Track A:", "Track B:")
_OTHER_DETECTION = "on another detection run than Track A"


class _TrackMergeConfig(ConfigWidget):
    def __init__(self):
        super().__init__()
        self._model_names: dict = {}
        self._config_form = ConfigForm(self)
        self._form = self._config_form.main
        intro = QLabel(field_meta(TrackMergeConfig, "source_specs")["help"])
        intro.setWordWrap(True)
        self._form.addRow(intro)
        self._pickers: list[UpstreamSpecPicker] = []
        self._rows: list[QComboBox] = []
        for label in _ROW_LABELS:
            picker = UpstreamSpecPicker("tracking")
            picker.combo.currentIndexChanged.connect(
                lambda _index: self.coverage_changed.emit()
            )
            self._pickers.append(picker)
            self._rows.append(picker.combo)
            self._form.addRow(
                field_label("track_merge", "source_specs", label), picker.combo
            )
        # {frozen spec: {recording_id: detection run id}} each row's tracking run read.
        self._reads: dict = {}
        self._rows[0].currentIndexChanged.connect(self._sync_lineage)

    def on_experiment_set(self, experiment) -> None:
        # A run record stores a model *id*; a picker has to show the name (§11).
        self._model_names = experiment.model_names() if experiment is not None else {}

    def resolve_eligible_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md): every recording eligible for tracking, not
        just the checked ones. Checking recordings one at a time otherwise locks this
        picker to whichever recording happened to be checked first — its own tracks are
        the only ones ``spec_groups`` would ever see, so every other recording's would
        show as "not in the selected run" even when it actually has a matching pair."""
        with read_cache.pinned():
            groups = spec_groups(recordings, "tracking", self._model_names)
            reads = spec_reads(recordings, "tracking", groups, "detection_run_id")
        return {"groups": groups, "reads": reads, "total": len(recordings)}

    def apply_eligible_recordings(self, recordings: list, resolved: dict) -> None:
        self._reads = dict(resolved.get("reads") or {})
        for picker in self._pickers:
            picker.apply(resolved["groups"], resolved["total"])
            if picker.combo.count() == 0:
                picker.combo.addItem("(no tracking run in this experiment)", None)
        self._sync_lineage()

    def _sync_lineage(self, *_args) -> None:
        """Offer the later rows only the tracking runs on Track A's detection run, recording
        by recording: the stage refuses sources built from different detection runs."""
        first = self._pickers[0]
        spec = first.spec()
        wanted = {} if spec is None else self._reads.get(run_spec.freeze_spec(spec), {})
        fits = None if spec is None else lineage_fits(self._reads, wanted)
        for picker in self._pickers[1:]:
            picker.restrict(fits, _OTHER_DETECTION)

    def covered_recording_ids(self) -> set[str] | None:
        """Recordings that have **every** selected source's tracking spec — a recording
        missing one would otherwise be launched and immediately fail `validate()`."""
        narrowed = None
        for picker in self._pickers:
            ids = picker.covered()
            if ids is None:
                continue
            narrowed = ids if narrowed is None else narrowed & ids
        return narrowed

    def config(self) -> dict:
        return {"source_specs": [picker.spec() for picker in self._pickers]}

    def validation_error(self) -> str | None:
        if any(picker.none_fit() for picker in self._pickers):
            return (
                "No other tracking run is on Track A's detection run: choose another "
                "Track A, or track the other animal on that detection run first."
            )
        if any(picker.spec() is None for picker in self._pickers):
            return "Pick a tracking run for every source — run tracking first."
        specs = [picker.spec() for picker in self._pickers]
        if len(specs) != len({run_spec.freeze_spec(s) for s in specs}):
            return "The same tracking run is picked more than once."
        return None


@register_module_ui
class TrackMergeModuleUI(ModuleUI):
    module_id = "track_merge"

    def config_widget(self) -> ConfigWidget:
        return _TrackMergeConfig()

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
