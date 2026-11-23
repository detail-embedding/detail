"""Comparison GUI (§24): a list page, a launch dialog, and a saved result you can read.

A comparison answers *does this differ between these groups?* — it has worked from the CLI since
#213 and had no screen, which put the numbers most likely to end up in a paper behind a terminal.

**Static on purpose.** `docs/gui.md` deferred this screen because a comparison "is read by
scrubbing between groups and time windows". That is true of a *live* screen; §24 makes a saved
comparison "a saved object holding its outputs, not a transient view", and a saved record's
groups, windows and aggregation path are fixed. Scrubbing belongs to §21's DIFFERENCES tab, which
runs a *new* comparison rather than re-reading this one.

Two things differ from the removed `gui/benchmark.py`, which this otherwise mirrored:

- **The bar is a fraction, not a count.** `run_comparison` reports `(stage, fraction, detail)`
  and emits `0.0 -> 1.0` *per stage*, so feeding its fraction straight to a bar walks it back to
  zero four times. `_StageBar` folds the stage's index in, which is the same thing
  `workers.processing_progress_fraction` does one layer up.
- **There is no fixed figure list.** A benchmark's figure names were a driver's static tuple; a
  comparison writes one figure per *measure*, named after it (`rate__rearing`). So the page reads
  what the export actually wrote out of the returned manifest rather than asking a registry.
"""

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detail.analysis import cohort, compare, comparison_report, node_measures
from detail.analysis import location_measures
from detail.storage import location_export
from detail.analysis.measures import MEASURES
from detail.configs.comparison import ComparisonConfig
from detail.configs.windows import WindowSpec
from detail.core import atomic
from detail.gui import sizing
from detail.gui import status_color
from detail.gui.dialog import Dialog
from detail.gui.lifetime import discard
from detail.gui.notes import NotesBox
from detail.gui.status_color import WARN, set_label_color
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    choice_combo,
    field_meta,
    field_widget,
    set_any_field_value,
)
from detail.gui.forms import ConfigForm
from detail.gui.busy import run_busy
from detail.gui.help_ui import help_button, label_with
from detail.gui.pages import (
    MONO,
    IndexTable,
    _heading,
    _divider,
    _fit_table_height,
    _heading,
    _index_caption,
    _instructions,
    _open_local_path,
    fit_list_height,
)
from detail.gui.recording_items import (
    fill_recording_list,
    offer_recordings,
    select_recording_ids,
    selected_recording_ids,
)
from detail.gui.tables import enable_sorting, give_slack_to, make_columns_resizable
from detail.gui.tables import row_id
from detail.gui.workers import StageProgressWorker
from detail.storage import behavior_output
from detail.storage import metadata as metadata_store


def _stages() -> tuple[str, ...]:
    """`run_comparison`'s stages, in the order it emits them — **from the runner**, not retyped.

    Used only to fold a per-stage fraction into a whole-run one; the names themselves come back
    on the callback. A local copy would mis-fold the bar the day a stage was renamed or
    reordered, and nothing would fail: the bar still moves, just to the wrong place.

    Read inside the function rather than imported at module scope. `detail.api.comparison` pulls
    the analysis and storage layers in, and a GUI module that drags them into *import* time
    changes what is loaded when — which is how a pending `BackgroundCall` in an unrelated screen
    started landing in the next test instead of its own. The same reason `comparison_report` is
    imported inside the functions that draw.
    """
    from detail.api.comparison import STAGES

    return STAGES


#: Group-by keys that are not metadata columns — taken from the backend rather than re-declared,
#: because a second copy is a second thing to keep in step. `window_id` is what makes "first third
#: vs last third" the same operation as "drug A vs B" rather than a separate feature (§24).
#:
#: `subject` is deliberately **not** offered as an aggregation path. It is a per-recording *track
#: index* (`core/utils.check_track_is_unambiguous`), `0` in every experiment that exists, so
#: "average within subject" reads as "average within animal" and means "average within track
#: slot". It stays available to compare across, where `describe_columns` shows it is constant.
_ROW_KEYS = compare.ROW_KEYS

#: What each row key is, since `metadata.csv` cannot say — they are produced by the measure pass.
_ROW_KEY_NOTE = {
    "subject": "the tracked animal within a recording",
    "window_id": "the time window, once you choose one below",
}


def _checkable(text: str, *, data: str | None = None) -> QListWidgetItem:
    item = QListWidgetItem(text)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(Qt.CheckState.Unchecked)
    item.setData(Qt.ItemDataRole.UserRole, data if data is not None else text)
    return item


def _checked(listwidget: QListWidget) -> list[str]:
    return [
        listwidget.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(listwidget.count())
        if listwidget.item(i).checkState() == Qt.CheckState.Checked
    ]


def _set_checked(listwidget: QListWidget, names: set[str]) -> None:
    """Check the rows naming ``names`` — skipping any the list has disabled.

    A disabled row is one the chosen source cannot answer, with the reason in its label. Seeding
    a comparison from one artifact into a form pointed at another could check such a row, so the
    form asked for a measure it had just finished explaining it could not compute.
    """
    for i in range(listwidget.count()):
        item = listwidget.item(i)
        if not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            continue  # its state is the list's to say: unticked, or ticked and fixed
        checked = item.data(Qt.ItemDataRole.UserRole) in names
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )


def _offered(listwidget: QListWidget) -> list[tuple[str, str]]:
    """``(data, text)`` for each row: what a list offers, to rebuild it only when that changes.

    A list rebuilt on every refresh is rebuilt inside its own ``itemChanged`` when ticking a
    row refreshes the form: the row that emitted the signal is deleted in its handler.
    """
    return [
        (listwidget.item(i).data(Qt.ItemDataRole.UserRole), listwidget.item(i).text())
        for i in range(listwidget.count())
    ]


def _source_label(experiment, run_id: str) -> str:
    """`an_03 · query · 12 recordings · 2026-08-21` — what every other run picker in this app
    shows (`modules/_common.run_label`), instead of a bare id.

    Read from `analysis.yaml` only. The behavior *names* live in `behavior_output.yaml`, whose
    reader parses both artifact CSVs — too much to pay per dropdown entry, for a label.
    """
    try:
        record = experiment.analysis_record(run_id) or {}
    except (FileNotFoundError, KeyError, ValueError):
        return run_id
    bits = [run_id, record.get("module_id")]
    recordings = record.get("recording_ids") or []
    if recordings:
        bits.append(f"{len(recordings)} recording(s)")
    created = (record.get("created_at") or "")[:10]
    bits.append(created or None)
    return "  ·  ".join(str(b) for b in bits if b)


def _measurable_runs(experiment) -> list[str]:
    """Analysis runs carrying a §19 behavior artifact — the only ones there is anything to
    compare.

    Discovery is by scanning for the artifact, not by filtering the run index: the question is
    "what can be compared", and the artifact is the thing that answers it. A producer added later
    appears here by emitting one, with nothing to register.
    """
    return behavior_output.discover_behavior_outputs(Path(experiment.dir))


#: The config fields that say how a location-export run is read into measures.
_LOCATION_FIELDS = (
    "location_sample_fps",
    "location_smooth_s",
    "location_moving_threshold",
    "location_jump_speed",
)


def _location_runs(experiment) -> list[str]:
    """The runs that wrote location tables: what the locations source can compare.

    Found by scanning for the tables, like the behavior results beside them, and not by
    reading every run's record, which is a round trip per run on the UI thread. A run that is
    still writing is listed too; launching over it is refused, saying it is not finished.
    """
    return location_export.discover_location_exports(Path(experiment.dir))


def _comparable_runs(experiment) -> list[str]:
    """Every run a comparison can be made over, in id order: the runs with a behavior
    artifact and the finished location exports. Which sources each one offers is the
    *Measure* picker's question (``_sync_measure_source``)."""
    return sorted(set(_measurable_runs(experiment)) | set(_location_runs(experiment)))


class _StageBar:
    """Fold `run_comparison`'s per-stage fraction into one that only goes forward.

    Each stage reports `0.0` then `1.0`, so a bar driven by the raw fraction resets at every
    stage boundary and reads as four failed attempts rather than one run.
    """

    def __init__(self, emit) -> None:
        self._emit = emit

    def __call__(self, stage: str, fraction: float, detail: str = "") -> None:
        stages = _stages()
        try:
            index = stages.index(stage)
        except ValueError:  # a stage added later still moves the bar, just coarsely
            index = len(stages) - 1
        self._emit(stage, (index + max(0.0, min(1.0, fraction))) / len(stages), detail)


class ComparisonLaunchDialog(Dialog):
    """Configure and run a comparison.

    **Every choice is shown with what is in it.** The first version of this screen offered a
    list of bare column names, and a user reading it could not tell what they were selecting,
    what the options were, or how to decide between them. Both
    questions are answerable from `metadata.csv` alone — which columns hold what, and which
    animal appears more than once — so `analysis/cohort.py` answers them here, before anything
    is measured. That also retires the special cases: a constant column like `subject` and a
    per-recording column like `date` both simply *say* what they hold and read as useless.

    **The unit of analysis is asked only when it arises.** §24 forbids defaulting it, because
    averaging each animal's recordings first and counting every recording separately give
    different answers — but that is only true when some animal contributed more than one
    recording. When nobody did, there is nothing to decide and the screen says so instead of
    asking. When somebody did, the question comes with the numbers that make it concrete and
    with the inferred animal column pre-selected, named where the user can see it and change it.
    """

    fits_content = True  # a settings form: show it all (Dialog.fits_content)

    def __init__(self, experiment, parent=None, seed: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("New comparison")
        self._experiment = experiment
        self.comparison = None  # set on a successful run
        self._worker: StageProgressWorker | None = None
        self._metadata = experiment.recording_metadata
        self._subject_column: str | None = None
        #: ``{(run, source, snapshot): recordings it covers, or None}``, read once per choice.
        self._covered_by: dict[tuple, list[str] | None] = {}

        layout = self.body()
        form = ConfigForm(layout)

        self._source = QComboBox()
        #: The location-export runs among them: what decides which sources a run offers.
        self._location_runs = set(_location_runs(experiment))
        for run_id in _comparable_runs(experiment):
            self._source.addItem(_source_label(experiment, run_id), run_id)
        # The newest, not the oldest. `discover_behavior_outputs` returns ascending id order and
        # a combo opens on index 0, so a user with twenty analyses landed on `an_01`. Ascending
        # order is kept — every other run picker in the app uses it — and the selection moves.
        self._source.setCurrentIndex(self._source.count() - 1)
        self._source.currentIndexChanged.connect(lambda *_: self._on_source_changed())
        form.addRow(
            label_with("Result to compare:", help_button("comparison-source")),
            self._source,
        )

        # What is measured: the run's labelled behaviors, or every node of a clustering run's
        # tree, named or not — the difference screen over clusters nobody has labelled (§24
        # *The node source*). Offered, greyed with the reason, when the run has no node cache.
        self._measure_source = QComboBox()
        self._measure_source.addItem("Labelled behaviors", "labels")
        self._measure_source.addItem("Every cluster (named or not)", "nodes")
        # Distance, speed and body shape, read from a location-export run (§24 *The
        # locations source*). Offered for those runs and greyed, with the reason, for others.
        self._measure_source.addItem("Movement and body shape", "locations")
        self._measure_source.currentIndexChanged.connect(
            lambda *_: self._on_measure_source_changed()
        )
        form.addRow(
            label_with("Measure:", help_button("comparison-measure")),
            self._measure_source,
        )

        self._columns = list(metadata_store.columns(Path(experiment.dir))) + list(
            _ROW_KEYS
        )
        self._group_by = QListWidget()
        self._group_by.itemChanged.connect(lambda *_: self._refresh())
        form.addRow(
            label_with("Compare across:", help_button("comparison-groups")),
            self._group_by,
        )

        # --- the unit of analysis ------------------------------------------- #
        self._unit_note = QLabel("")
        self._unit_note.setWordWrap(True)
        self._per_recording = QRadioButton("Count every recording separately")
        self._per_column = QRadioButton("Average within these first:")
        self._per_recording.toggled.connect(lambda *_: self._refresh())
        self._per_column.toggled.connect(lambda *_: self._refresh())
        self._aggregate_to = QListWidget()
        self._aggregate_to.itemChanged.connect(lambda *_: self._refresh())
        # One container, because hiding a `QFormLayout` row's widgets leaves the row — and the
        # whole point is that a cohort with no repeats sees nothing here, not a gap where a
        # question used to be.
        self._unit_box = QWidget()
        unit_layout = QVBoxLayout(self._unit_box)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.addWidget(
            label_with("One observation is:", help_button("comparison-unit"))
        )
        unit_layout.addWidget(self._per_recording)
        unit_layout.addWidget(self._per_column)
        unit_layout.addWidget(self._aggregate_to)
        form.addRow(self._unit_note)
        form.addRow(self._unit_box)

        # Two values of the one chosen column to contrast — the one lever that changes which
        # design runs (§24 *Designs*). Shown only when exactly one metadata column is chosen.
        self._levels = QListWidget()
        #: The ticked values in the order they were ticked: the contrast reads as the second
        #: against the first, and a list widget alone remembers only which, not when.
        self._levels_order: list[str] = []
        #: The column whose values the list holds: another column's ticks mean nothing here.
        self._levels_column: str | None = None
        self._levels.itemChanged.connect(lambda *_: self._on_levels_changed())
        self._levels_box = QWidget()
        levels_layout = QVBoxLayout(self._levels_box)
        levels_layout.setContentsMargins(0, 0, 0, 0)
        levels_layout.addWidget(
            label_with(
                "Contrast these two (optional):", help_button("comparison-levels")
            )
        )
        levels_layout.addWidget(self._levels)
        form.addRow(self._levels_box)

        # The design the choices above resolve to, before the run: the same function the run
        # uses, so the preview and the record cannot disagree.
        self._design_note = QLabel("")
        self._design_note.setWordWrap(True)
        form.addRow(label_with("Design:", help_button("comparison-design")))
        form.addRow(self._design_note)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        form.addRow(self._summary)

        # --- advanced -------------------------------------------------------- #
        self._recordings = QListWidget()
        self._recordings.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        fill_recording_list(self._recordings, list(experiment.recordings))
        fit_list_height(self._recordings)
        self._recordings.itemSelectionChanged.connect(self._on_recordings_changed)
        form.addRow(
            label_with(
                "Recordings (all if none):", help_button("comparison-recordings")
            ),
            self._recordings,
            advanced=True,
        )

        self._snapshot = QComboBox()
        self._fill_snapshots()
        # A snapshot is a frozen copy of the artifact, and may cover other recordings.
        self._snapshot.currentIndexChanged.connect(lambda *_: self._sync_cohort())
        form.addRow(
            label_with("Read snapshot:", help_button("comparison-snapshot")),
            self._snapshot,
            advanced=True,
        )

        # From the model, not typed here: `choice_combo`'s own docstring names this as the way
        # three forms each grow their own copy of one model's option list, and a fifth window
        # kind added to `WindowSpec` would otherwise be unreachable from this dialog.
        self._windows = choice_combo(WindowSpec, "kind")
        self._windows.currentTextChanged.connect(self._sync_window_fields)
        form.addRow(
            label_with("Windows:", help_button("comparison-windows")),
            self._windows,
            advanced="Windows",
        )
        self._window_fields = {
            name: field_widget(WindowSpec, name)
            for name in ("every_s", "parts", "length_s", "step_s")
        }
        for name, widget in self._window_fields.items():
            form.addRow(
                label_with(
                    f"{field_meta(WindowSpec, name)['label']}:",
                    help_button("comparison-windows"),
                ),
                widget,
                advanced="Windows",
            )

        self._measures = QListWidget()
        #: The measures the user ticked, kept while *Every cluster* shows all four fixed, so
        #: switching back to labels restores the choice rather than the fixed four.
        self._measures_chosen: set[str] = set()
        self._measures_fixed = False
        self._fill_measures()
        form.addRow(
            label_with("Measures (all if none):", help_button("comparison-measures")),
            self._measures,
            advanced=True,
        )

        self._draws = field_widget(ComparisonConfig, "draws")
        self._seed = field_widget(ComparisonConfig, "seed")
        self._ranking_fields = {
            # A `Literal` field is a combo (§24 *The statistic*); the rest are numbers and
            # a bool, which `field_widget` renders.
            "statistic": choice_combo(ComparisonConfig, "statistic"),
            **{
                name: field_widget(ComparisonConfig, name)
                for name in (
                    "presence_s",
                    "pseudo_count_s",
                    "min_units_fraction",
                    "candidate_floor_s",
                    "suppress_nesting",
                )
            },
        }
        for name, widget in (
            ("draws", self._draws),
            ("seed", self._seed),
            *self._ranking_fields.items(),
        ):
            form.addRow(
                label_with(
                    f"{field_meta(ComparisonConfig, name)['label']}:",
                    help_button("comparison-floor"),
                ),
                widget,
                advanced="Difference ranking",
            )

        # How a location-export run is read into measures. Built from the model, like the
        # ranking knobs, and enabled only while *Movement and body shape* is the measure.
        self._location_fields = {
            name: any_field_widget(ComparisonConfig, name) for name in _LOCATION_FIELDS
        }
        rate = self._location_fields["location_sample_fps"]
        rate.setSpecialValueText("every row")
        # An optional box opens on "unset", and unset here is a real choice (every row) that
        # is not the model's default: seed it, or every comparison launched from this form
        # reads the run at another rate than one launched from the CLI.
        set_any_field_value(
            ComparisonConfig,
            "location_sample_fps",
            rate,
            ComparisonConfig.model_fields["location_sample_fps"].default,
        )
        for name, widget in self._location_fields.items():
            # Off until *Movement and body shape* is the measure (`_on_measure_source_changed`).
            widget.setEnabled(False)
            form.addRow(
                label_with(
                    f"{field_meta(ComparisonConfig, name)['label']}:",
                    help_button("comparison-locations"),
                ),
                widget,
                advanced="Locations",
            )

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run")
        self._cancel = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._launch)
        self._buttons.rejected.connect(self.reject)

        self._sync_window_fields(self._windows.currentText())
        self._sync_measure_source()
        self._sync_cohort()
        form.finish()
        if seed:
            self._apply_seed(seed)
        # Last, so it sees the finished layout: the lists are capped by `fit_list_height` in the
        # fills above, and before that they still claim to expand.
        self.set_buttons(self._buttons)

    def _on_source_changed(self) -> None:
        self._fill_snapshots()
        self._sync_measure_source()
        self._fill_measures()
        self._sync_cohort()

    def _covered_recordings(self) -> list[str] | None:
        """The recordings the chosen run covers for the chosen measure, or ``None`` when that
        cannot be read (the run then refuses in its own words).

        The rule the run applies (``api.comparison.source_recordings``), asked here so every
        preview below counts the recordings the run will compare. Read once per choice: a
        coverage table, a run record or a node cache's id list, one small read each.
        """
        run_id = self.source_run_id()
        if not run_id:
            return None
        source = self.measure_source()
        snapshot = self._snapshot.currentData() if source == "labels" else None
        key = (run_id, source, snapshot)
        if key not in self._covered_by:
            from detail.api.comparison import source_recordings

            try:
                self._covered_by[key] = source_recordings(
                    self._experiment, run_id, source=source, snapshot=snapshot
                )
            except (OSError, ValueError):
                self._covered_by[key] = None
        return self._covered_by[key]

    def _sync_cohort(self) -> None:
        """Offer only the recordings the chosen run covers, then describe every column over
        them: what *Compare across* and *Average within* list, and the counts beside them.
        """
        self._recordings.blockSignals(True)
        offer_recordings(
            self._recordings,
            self._covered_recordings(),
            f"not in {self.source_run_id()}",
        )
        self._recordings.blockSignals(False)
        self._on_recordings_changed()

    def _on_recordings_changed(self) -> None:
        self._fill_group_by()
        self._refresh()

    def _on_measure_source_changed(self) -> None:
        self._fill_measures()
        nodes = self.measure_source() == "nodes"
        locations = self.measure_source() == "locations"
        for widget in self._location_fields.values():
            widget.setEnabled(locations)
        # The node source measures the whole recording (§24): the window picker is greyed
        # with that set, not hidden, so the constraint reads as one.
        if nodes and self._windows.findText("whole") >= 0:
            self._windows.setCurrentText("whole")
        self._windows.setEnabled(not nodes)
        self._windows.setToolTip(
            "Every cluster is measured over the whole recording." if nodes else ""
        )
        # Nor from a pinned copy: a snapshot pins the labelled artifact, and every cluster
        # is measured from the run's current node cache.
        if nodes or locations:
            self._snapshot.setCurrentIndex(0)
        self._snapshot.setEnabled(not (nodes or locations))
        self._snapshot.setToolTip(
            "Every cluster is measured from the run's current node cache."
            if nodes
            else (
                "A location-export run is never rewritten, so it has no snapshots."
                if locations
                else ""
            )
        )
        # The same run covers other recordings under another measure: a clustering run's
        # node cache, say, against its labelled artifact.
        self._sync_cohort()

    def measure_source(self) -> str:
        return self._measure_source.currentData() or "labels"

    def _node_cache_reason(self) -> str | None:
        """Why the chosen run's clusters cannot be screened, or ``None`` when they can."""
        run_id = self.source_run_id()
        if not run_id:
            return "no run chosen"
        try:
            run = self._experiment.authored_run(run_id)
        except (OSError, ValueError, KeyError):
            run = None
        if run is None or not hasattr(run, "node_view"):
            return "not a clustering run"
        if not run.has_node_cache():
            return "this run has no node cache — rebuild it from its workspace"
        return None

    def _source_reasons(self) -> dict[str, str | None]:
        """``{source: why the chosen run cannot offer it}``, ``None`` where it can.

        A location-export run offers movement and body shape and nothing else; every other
        run offers its labelled behaviors, and its clusters when it has a node cache.
        """
        if self.source_run_id() in self._location_runs:
            no_behaviors = "a location-export run holds no behaviors"
            return {"labels": no_behaviors, "nodes": no_behaviors, "locations": None}
        return {
            "labels": None,
            "nodes": self._node_cache_reason(),
            "locations": "pick a location-export run under Result to compare",
        }

    def _sync_measure_source(self) -> None:
        """Grey out each measure the chosen run cannot offer, with the reason, and move the
        selection off one that just became unavailable."""
        reasons = self._source_reasons()
        model = self._measure_source.model()
        for index in range(self._measure_source.count()):
            reason = reasons.get(self._measure_source.itemData(index))
            item = model.item(index)
            if item is not None:
                item.setEnabled(reason is None)
                item.setToolTip(reason or "")
        if reasons.get(self.measure_source()) is not None:
            offered = [
                index
                for index in range(self._measure_source.count())
                if reasons.get(self._measure_source.itemData(index)) is None
            ]
            if offered:
                self._measure_source.setCurrentIndex(offered[0])

    def _fill_snapshots(self) -> None:
        """Which pinned copy of the source's artifact to read (§21).

        Left unset a comparison pins whatever the run holds now and records the id it got, which
        is the right default. But a saved comparison stores the *resolved* id, so seeding a new
        one from it has to be able to say "that copy" — otherwise "same source, different
        grouping" quietly compares a re-cut artifact and the two are not comparable.
        """
        from detail.storage import analysis_runs as analysis_store
        from detail.storage import snapshots as snapshots_store

        wanted = self._snapshot.currentData() if self._snapshot.count() else None
        # Quiet while it is rebuilt: each step would re-read the cohort of a half-chosen
        # source. The caller syncs it once, after.
        self._snapshot.blockSignals(True)
        self._snapshot.clear()
        self._snapshot.addItem("Current artifact", None)
        run_id = self.source_run_id()
        if run_id:
            run_dir = analysis_store.analysis_dir(Path(self._experiment.dir), run_id)
            for snapshot_id in snapshots_store.discover_snapshots(run_dir):
                self._snapshot.addItem(snapshot_id, snapshot_id)
        index = self._snapshot.findData(wanted)
        self._snapshot.setCurrentIndex(max(index, 0))
        self._snapshot.blockSignals(False)

    def _apply_seed(self, seed: dict) -> None:
        """Pre-fill from a saved comparison's record — "same source, different grouping" is the
        loop people run, and re-entering six fields to change one is the friction.

        `config()` returns the shape the record stores, so this is a round-trip. It seeds a
        *new* comparison: §24 makes a saved one immutable, like `launch.py::_apply_seed` does
        for runs.
        """
        index = self._source.findData(seed.get("source_run_id"))
        if index >= 0:
            self._source.setCurrentIndex(index)
        self._fill_snapshots()
        snapshot = self._snapshot.findData(seed.get("source_snapshot"))
        if snapshot >= 0:
            self._snapshot.setCurrentIndex(snapshot)
        select_recording_ids(self._recordings, seed.get("recordings") or [])
        self._draws.setValue(int(seed.get("draws", self._draws.value())))
        self._seed.setValue(int(seed.get("seed", self._seed.value())))
        self._fill_group_by()
        _set_checked(self._group_by, set(seed.get("group_by") or []))
        self._refresh()
        # Only a seed that *says* what one observation is may set it. A record always does
        # (`[]` is its real answer, every recording one point); the dock's "New screen…" seed
        # names the run and the source and nothing else, and reading its silence as `[]`
        # launched every screen from the dock per recording -- the between design whose floor
        # the node source cannot clear (§24 *Designs*) -- over the form's own default of
        # averaging within the inferred subject column.
        if self._nested() and "aggregate_to" in seed:
            path = list(seed.get("aggregate_to") or [])
            self._per_column.setChecked(bool(path))
            self._per_recording.setChecked(not path)
            if path:
                _set_checked(self._aggregate_to, set(path))
        windows = seed.get("windows") or {}
        if windows.get("kind"):
            self._windows.setCurrentText(windows["kind"])
        for name, widget in self._window_fields.items():
            if windows.get(name) is not None:
                widget.setValue(windows[name])
        self._fill_measures()
        _set_checked(self._measures, set(seed.get("measures") or []))
        self._apply_design_fields(seed)
        self._refresh()

    def _fill_measures(self) -> None:
        """Offer the measures the *chosen source* can answer, and grey out the rest with why.

        Not every measure applies to every artifact: `Transitions` needs behaviors that cannot
        overlap, and `Time in unnamed clusters` needs an artifact that can have unnamed clusters
        — which a query run cannot. Offering them anyway produced a comparison row of zeros that
        said nothing, which is what a user hit.

        Greyed rather than hidden ("grey out, don't hide"), with the reason attached, so it reads
        as "not for this run" rather than "not supported" — the shape `gui/model_availability.py`
        settled on for the same problem with model pickers.
        """
        from detail.analysis.measures import unmet_condition

        manifest = self._source_manifest()
        nodes = self.measure_source() == "nodes"
        if not self._measures_fixed:
            self._measures_chosen = set(_checked(self._measures))
        checked = self._measures_chosen
        self._measures_fixed = nodes
        self._measures.blockSignals(True)
        self._measures.clear()
        if self.measure_source() == "locations":
            # Another registry: what a location-export run can be measured for.
            for measure_id, spec in location_measures.MEASURES.items():
                item = _checkable(spec.label, data=measure_id)
                item.setToolTip(spec.description)
                if measure_id in checked:
                    item.setCheckState(Qt.CheckState.Checked)
                self._measures.addItem(item)
            self._measures.blockSignals(False)
            fit_list_height(self._measures)
            return
        for measure_id, spec in MEASURES.items():
            why = unmet_condition(spec, manifest) if manifest is not None else None
            if nodes and measure_id not in node_measures.MEASURES:
                why = "not measured over clusters"
            elif nodes:
                why = None
            text = f"{spec.label}   —   {why}" if why else spec.label
            item = _checkable(text, data=measure_id)
            item.setToolTip(spec.description)
            if why:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            elif nodes:
                # Every cluster is measured by all four, always: shown ticked and fixed.
                item.setCheckState(Qt.CheckState.Checked)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(
                    f"{spec.description} Every cluster is measured by all four."
                )
            elif measure_id in checked:
                item.setCheckState(Qt.CheckState.Checked)
            self._measures.addItem(item)
        self._measures.blockSignals(False)
        fit_list_height(self._measures)

    def _source_manifest(self) -> dict | None:
        """The chosen source artifact's manifest, or `None` when it cannot be read.

        Manifest only — `read_behavior_output` parses both artifact CSVs, which is far too much
        to pay on the UI thread for a picker.
        """
        import yaml

        run_id = self.source_run_id()
        if not run_id:
            return None
        path = (
            Path(self._experiment.dir)
            / "analysis"
            / run_id
            / behavior_output.MANIFEST_FILENAME
        )
        if not path.is_file():
            return None
        try:
            return yaml.safe_load(atomic.read_text(path)) or {}
        except (OSError, yaml.YAMLError):
            return None

    # -- what is in each column ------------------------------------------------ #
    def _selected_recordings(self) -> list[str]:
        """The recordings the comparison will compare: those ticked, else every one the chosen
        run covers. Every preview on this form is described over these."""
        covered = self._covered_recordings()
        if covered is None:
            covered = self._recordings_all()
        chosen = set(selected_recording_ids(self._recordings))
        return [r for r in covered if r in chosen] or list(covered)

    def _recordings_all(self) -> list[str]:
        """Every recording in the experiment, as the recordings list holds them."""
        return [
            self._recordings.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self._recordings.count())
        ]

    def _fill_group_by(self) -> None:
        """One row per column, labelled with the values it holds.

        A column that cannot define groups is **disabled with the reason in its label** rather
        than dropped — "grey out, don't hide", and a hidden option is one the user goes looking
        for. This is also what keeps `subject` from needing a special case: it is constant, so
        it says so.
        """
        recordings = self._selected_recordings()
        columns = [c for c in self._columns if c not in _ROW_KEYS]
        offered = []
        for summary in cohort.describe_columns(self._metadata, recordings, columns):
            refusal = summary.usable_for_grouping
            text = f"{summary.name}   —   {refusal or summary.describe()}"
            offered.append((summary.name, text, bool(refusal)))
        # The row keys come off the measures table, which does not exist until the run — so
        # there is nothing to describe and nothing to refuse. Saying where they come from beats
        # `describe_columns` reporting them, correctly but uselessly, as "no values recorded".
        for name in _ROW_KEYS:
            offered.append((name, f"{name}   —   {_ROW_KEY_NOTE[name]}", False))
        if [(name, text) for name, text, _ in offered] == _offered(self._group_by):
            return
        # Re-described when the recordings change (another run, another selection), keeping
        # what was ticked where it can still be: a column that now holds one value is greyed
        # and comes back unticked.
        already = set(_checked(self._group_by))
        self._group_by.blockSignals(True)
        self._group_by.clear()
        for name, text, refused in offered:
            item = _checkable(text, data=name)
            if refused:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._group_by.addItem(item)
        _set_checked(self._group_by, already)
        self._group_by.blockSignals(False)
        fit_list_height(self._group_by)

    def _fill_aggregate_to(self, group_by: list[str]) -> None:
        """The columns available to average within, with the inferred animal pre-checked."""
        recordings = self._selected_recordings()
        already = set(_checked(self._aggregate_to))
        candidates = [
            c for c in self._columns if c not in group_by and c not in _ROW_KEYS
        ]
        offered = [
            (summary.name, f"{summary.name}   —   {summary.describe()}")
            for summary in cohort.describe_columns(
                self._metadata, recordings, candidates
            )
        ]
        # `window_id` is offered to *compare across* but not to average within: averaging
        # through the windows is what `kind="whole"` already does, and the row key has no
        # values here to show for itself.
        if "window_id" not in group_by:
            offered.append(
                ("window_id", f"window_id   —   {_ROW_KEY_NOTE['window_id']}")
            )
        self._aggregate_to.blockSignals(True)
        if offered != _offered(self._aggregate_to):
            self._aggregate_to.clear()
            for name, text in offered:
                self._aggregate_to.addItem(_checkable(text, data=name))
        _set_checked(
            self._aggregate_to,
            already or ({self._subject_column} if self._subject_column else set()),
        )
        self._aggregate_to.blockSignals(False)
        fit_list_height(self._aggregate_to)

    # -- state ---------------------------------------------------------------- #
    def _sync_window_fields(self, kind: str) -> None:
        shown = {
            "absolute": ("every_s",),
            "normalized": ("parts",),
            "sliding": ("length_s", "step_s"),
        }.get(kind, ())
        for name, widget in self._window_fields.items():
            widget.setEnabled(name in shown)

    def _nested(self) -> bool:
        """Whether some animal contributed more than one of the selected recordings."""
        return self._subject_column is not None

    def validation_error(self) -> str | None:
        if not self.source_run_id():
            return (
                "Nothing to compare yet — run a behavior scan, a clustering or a location "
                "export first."
            )
        if self._covered_recordings() == []:
            return (
                f"{self.source_run_id()} has not finished a recording yet, so there is "
                "nothing in it to compare."
            )
        if not _checked(self._group_by):
            return "Pick at least one column to compare across."
        if self._nested():
            if not (self._per_recording.isChecked() or self._per_column.isChecked()):
                return (
                    "Say what one observation is — the two answers differ whenever animals "
                    "contribute unequal numbers of recordings."
                )
            if self._per_column.isChecked() and not _checked(self._aggregate_to):
                return "Pick the columns to average within, or count every recording separately."
        overlap = sorted(set(_checked(self._group_by)) & set(self.aggregate_to()))
        if overlap:
            return (
                f"{overlap} would both define the groups and be averaged through, so every "
                f"group would hold exactly one observation. Compare across it, or average "
                f"within it — not both."
            )
        ticked = _checked(self._levels) if not self._levels_box.isHidden() else []
        if ticked and len(ticked) != 2:
            return "Tick exactly two values to contrast, or none."
        refusal = self._design_refusal()
        if refusal:
            return refusal
        foreseen = self._foreseen_no_floor()
        if foreseen is not None and self.measure_source() == "nodes":
            from detail.api.comparison import screen_refusal

            return screen_refusal(foreseen)
        return None

    def levels(self) -> list[str] | None:
        """The two values to contrast, in the order ticked, or ``None`` for "let the column
        decide"."""
        if self._levels_box.isHidden():
            return None
        ticked = _checked(self._levels)
        if len(ticked) != 2:
            return None
        ordered = [v for v in self._levels_order if v in ticked]
        return ordered + [v for v in ticked if v not in ordered]

    def _on_levels_changed(self) -> None:
        ticked = _checked(self._levels)
        self._levels_order = [v for v in self._levels_order if v in ticked] + [
            v for v in ticked if v not in self._levels_order
        ]
        self._refresh()

    def _fill_levels(self, group_by: list[str]) -> None:
        """The chosen column's values, to tick two of — shown only with one metadata column."""
        recordings = self._selected_recordings()
        column = group_by[0] if len(group_by) == 1 else None
        if column is None or column in _ROW_KEYS:
            self._levels_box.setVisible(False)
            self._levels_column = None
            return
        summary = cohort.describe_columns(self._metadata, recordings, [column])[0]
        offered = [
            (value, f"{value}   —   {count} recording(s)")
            for value, count in sorted(summary.counts.items(), key=lambda kv: kv[0])
        ]
        if column != self._levels_column:
            self._levels_column = column
            self._levels_order = []
            already: set[str] = set()
        elif offered == _offered(self._levels):
            return  # a tick, not a new offer: the list stays, and so do its rows
        else:
            already = set(_checked(self._levels))
        self._levels.blockSignals(True)
        self._levels.clear()
        for value, text in offered:
            self._levels.addItem(_checkable(text, data=value))
        _set_checked(self._levels, already)
        self._levels.blockSignals(False)
        fit_list_height(self._levels)
        self._levels_box.setVisible(True)

    def _design_refusal(self) -> str | None:
        """Why the chosen column cannot resolve to a design yet, or ``None``."""
        group_by = _checked(self._group_by)
        recordings = self._selected_recordings()
        if not group_by or cohort.deferred_columns(
            self._metadata, recordings, group_by
        ):
            return None
        try:
            compare.resolve_design(
                self._metadata,
                recordings,
                group_by=group_by,
                aggregate_to=self.aggregate_to(),
                levels=self.levels(),
            )
        except compare.ComparisonError as e:
            return str(e)
        return None

    def _foreseen_no_floor(self):
        """Why this comparison could have no floor, knowable before it runs (§24), or ``None``.

        The one rule the run applies (``compare.no_floor_before_measuring``), asked here so a
        screen that would be refused is refused before Run rather than by the run (#39), and a
        labelled comparison that will say *No floor computed* says so in the preview.
        """
        group_by = _checked(self._group_by)
        recordings = self._selected_recordings()
        if not group_by or cohort.deferred_columns(
            self._metadata, recordings, group_by
        ):
            return None
        nodes = self.measure_source() == "nodes"
        try:
            return compare.no_floor_before_measuring(
                self._metadata,
                recordings,
                group_by=group_by,
                aggregate_to=self.aggregate_to(),
                levels=self.levels(),
                one_row_per_recording=nodes or self._windows.currentText() == "whole",
            )
        except compare.ComparisonError:
            return None  # `_design_refusal` carries that message

    def _design_text(self, recordings: list[str], group_by: list[str]) -> str:
        """What will run, in the sentence the record will carry (§24 *Designs*)."""
        if not group_by:
            return ""
        deferred = cohort.deferred_columns(self._metadata, recordings, group_by)
        if deferred:
            return f"Decided when it runs: {', '.join(deferred)} come from the result itself."
        try:
            design = compare.resolve_design(
                self._metadata,
                recordings,
                group_by=group_by,
                aggregate_to=self.aggregate_to(),
                levels=self.levels(),
            )
        except compare.ComparisonError as e:
            return f"\u26a0 {e}"
        return f"{design.kind} — {design.reason}"

    def source_run_id(self) -> str:
        return self._source.currentData() or ""

    def aggregate_to(self) -> list[str]:
        """The aggregation path. `[]` is a real answer — "every recording is one observation" —
        and it is what a cohort with no repeats gets without being asked."""
        if not self._nested():
            return []
        return _checked(self._aggregate_to) if self._per_column.isChecked() else []

    def _refresh(self) -> None:
        group_by = _checked(self._group_by)
        recordings = self._selected_recordings()
        self._subject_column = cohort.infer_subject_column(
            self._metadata, recordings, group_by=group_by, columns=self._columns
        )
        self._fill_aggregate_to(group_by)
        self._fill_levels(group_by)

        nested = self._nested()
        self._unit_box.setVisible(nested)
        self._aggregate_to.setVisible(nested and self._per_column.isChecked())
        if nested and not (
            self._per_recording.isChecked() or self._per_column.isChecked()
        ):
            self._per_column.setChecked(True)  # the defensible answer, pre-selected
        self._unit_note.setText(self._unit_text(recordings, group_by))
        self._summary.setText(self._summary_text(recordings, group_by))
        self._design_note.setText(self._design_text(recordings, group_by))

        error = self.validation_error()
        sizing.set_available(
            self._buttons.button(QDialogButtonBox.StandardButton.Ok),
            error is None,
            error or "",
        )
        self._status.setText(error or "")
        status_color.set_label_color(
            self._status, status_color.WARN if error else status_color.MUTED
        )

    def _unit_text(self, recordings: list[str], group_by: list[str]) -> str:
        """Why the question is being asked, in the register the one product that explains this
        at all uses. GraphPad Prism's guide: *"the replicates for each animal are closer to each
        other than to the values obtained from other animals."* Prism, JMP, SPSS and jamovi
        contain the words "pseudoreplication" or "unit of analysis" nowhere in any UI — so this
        is a gap to fill rather than a pattern to copy.
        """
        if not self._subject_column:
            return (
                f"Each of the {len(recordings)} recording(s) is one observation — no value of "
                f"any column covers more than one of them."
            )
        counts = cohort.describe_columns(
            self._metadata, recordings, [self._subject_column]
        )[0]
        most = max(counts.counts.values())
        return (
            f"{counts.n_values} value(s) of '{self._subject_column}' cover these "
            f"{len(recordings)} recordings, one of them {most} times. Recordings from one "
            f"animal are more alike than recordings from different animals, so that one would "
            f"outweigh the rest."
        )

    def _summary_text(self, recordings: list[str], group_by: list[str]) -> str:
        """What this comparison will produce, before it is run — the pattern
        removed `benchmark.py::_ClusterEvalMemberForm.summary` set: learning it from the result
        is too late."""
        if not group_by:
            return ""
        deferred = cohort.deferred_columns(self._metadata, recordings, group_by)
        if deferred:
            # Not previewable and not blank: `subject` and `window_id` come off the behavior
            # result's own rows and the window spec, so their groups do not exist until the
            # comparison runs. Reading them out of `metadata.csv` gave every recording the empty
            # string, which previewed as one group named "(blank)" holding the whole cohort — a
            # confident preview of a comparison that will produce something else entirely.
            names = ", ".join(deferred)
            return (
                f"Groups for {names} come from the behavior result itself, so their sizes are "
                f"known once the comparison runs — {len(recordings)} recording(s) go in."
            )
        # A recording with no value for a grouping column is out before anything is grouped
        # (section 24), so neither the groups nor the count below include it.
        outside = cohort.left_out(self._metadata, recordings, group_by)
        recordings = [r for r in recordings if r not in set(outside)]
        groups = cohort.group_preview(
            self._metadata,
            recordings,
            group_by=group_by,
            subject_column=self._subject_column,
        )
        levels = self.levels()
        if levels:
            # Two levels named: the other groups sit outside the contrast, and a preview that
            # still counted them would describe a comparison that will not run.
            groups = [g for g in groups if g.label in levels]
            recordings = recordings[: sum(g.n_recordings for g in groups)]
        head = f"{len(groups)} group(s) over {len(recordings)} recording(s):"
        lines = []
        covered = self._covered_recordings()
        total = len(self._recordings_all())
        if covered is not None and len(covered) < total:
            lines.append(
                f"    {self.source_run_id()} covers {len(covered)} of this experiment's "
                f"{total} recordings; only those can be compared."
            )
        for group in groups:
            bit = f"    {group.label} — {group.n_recordings} recording(s)"
            if group.n_subjects is not None:
                # "distinct" rather than a pluralized column name: `1 animal(s)` reads badly
                # and the count is exactly the thing the aggregation question turns on.
                bit += f", {group.n_subjects} distinct {self._subject_column}"
            lines.append(bit)
        thin = [g.label for g in groups if g.n_recordings < 2]
        if thin:
            lines.append(
                f"    ⚠ {', '.join(thin)} hold one recording each, so they have no spread."
            )
        foreseen = self._foreseen_no_floor()
        if foreseen is not None and self.measure_source() != "nodes":
            # A labelled comparison runs without a floor and is reported as measured; a
            # screen is refused instead (`validation_error`), so this line is not doubled.
            lines.append(
                f"    ⚠ No floor will be computed: {foreseen.reason}. {foreseen.remedy}"
            )
        if outside:
            # The same sentence the record will carry (§24), before the run rather than after.
            lines.append(
                "    "
                + comparison_report.left_out_sentence(
                    {"left_out": outside, "group_by": group_by},
                    self._experiment.recording_names(),
                )
            )
        return "\n".join([head, *lines])

    def load_config(self, config: dict) -> None:
        """Restore a previous comparison's settings — the inverse of :meth:`config`.

        ``source_run_id`` and ``source_snapshot`` are *selections*: they name a run and a frozen
        view of it, and the dialog re-asks. ``recordings`` is one too — it is the dialog's own
        list — so what comes back is how the comparison was *shaped*: what it grouped by, what
        it aggregated to, which measures it took, its window and its null.
        """
        _set_checked(self._group_by, set(config.get("group_by") or []))
        _set_checked(self._measures, set(config.get("measures") or []))
        # `aggregate_to` is a *path*, and `[]` is a real answer ("every recording is one
        # observation"), so the per-column box has to be driven from whether the stored path is
        # empty rather than left wherever the form put it.
        aggregate_to = config.get("aggregate_to") or []
        self._per_column.setChecked(bool(aggregate_to))
        _set_checked(self._aggregate_to, set(aggregate_to))
        windows = config.get("windows") or {}
        kind = windows.get("kind")
        if kind and self._windows.findText(kind) >= 0:
            self._windows.setCurrentText(kind)
        for name, widget in self._window_fields.items():
            if windows.get(name) is not None:
                widget.setValue(windows[name])
        for key, widget in (("draws", self._draws), ("seed", self._seed)):
            if config.get(key) is not None:
                widget.setValue(int(config[key]))
        self._apply_design_fields(config)

    def _apply_design_fields(self, config: dict) -> None:
        """The source, the levels and the ranking knobs — shared by `load_config` and
        `_apply_seed`, so a field restored by one is restored by the other."""
        for key, widget in self._ranking_fields.items():
            if config.get(key) is not None:
                set_any_field_value(ComparisonConfig, key, widget, config[key])
        # `in`, not `is not None`: an unset rate ("every row") is a value to restore.
        for key, widget in self._location_fields.items():
            if key in config:
                set_any_field_value(ComparisonConfig, key, widget, config[key])
        source = config.get("source")
        if source and self._measure_source.findData(source) >= 0:
            self._measure_source.setCurrentIndex(self._measure_source.findData(source))
        levels = config.get("levels") or []
        if levels:
            self._refresh()  # fills the levels list for the restored column
            _set_checked(self._levels, set(levels))
            self._levels_order = list(levels)
            self._refresh()

    def config(self) -> dict:
        """Every field of `ComparisonConfig`, so the form and the record hold the same comparison.

        Guarded field-by-field over `ComparisonConfig.model_fields`
        (`tests/gui/test_comparison_gui.py`): a dict written out by hand is complete until the day
        a field is added, and nothing fails when it stops being — the run just quietly uses the
        default for whatever was dropped.
        """
        return {
            "source_run_id": self.source_run_id(),
            "source_snapshot": (
                None
                if self.measure_source() in ("nodes", "locations")
                else self._snapshot.currentData()
            ),
            "source": self.measure_source(),
            "group_by": _checked(self._group_by),
            "aggregate_to": self.aggregate_to(),
            "levels": self.levels(),
            "recordings": selected_recording_ids(self._recordings) or None,
            "measures": (
                None
                if self.measure_source() == "nodes"
                else _checked(self._measures) or None
            ),
            "windows": {
                "kind": self._windows.currentText(),
                **{
                    name: widget.value() for name, widget in self._window_fields.items()
                },
            },
            "draws": self._draws.value(),
            "seed": self._seed.value(),
            **{
                name: any_field_value(ComparisonConfig, name, widget)
                for name, widget in self._ranking_fields.items()
            },
            **{
                name: any_field_value(ComparisonConfig, name, widget)
                for name, widget in self._location_fields.items()
            },
        }

    # -- launch ---------------------------------------------------------------- #
    def _launch(self) -> None:
        error = self.validation_error()
        if error:
            QMessageBox.warning(self, "New comparison", error)
            return
        config = self.config()
        # Ok goes down, Cancel stays up and becomes Stop. Disabling the whole box left Esc as
        # the only way out, and Esc ran `reject()` -- which closed the dialog and left the
        # worker measuring into a window nobody could see.
        sizing.set_available(
            self._buttons.button(QDialogButtonBox.StandardButton.Ok),
            False,
            "Running — use Stop to cancel",
        )
        self._cancel.setText("Stop")
        self._progress.setValue(0)
        self._progress.show()
        self._status.setText("Measuring…")
        status_color.set_label_color(self._status, status_color.MUTED)

        worker = StageProgressWorker(
            lambda emit: self._experiment.run_comparison(
                config,
                progress=_StageBar(emit),
                stop_check=worker.stop_requested,
            ),
            receiver=self,
        )
        self._worker = worker
        worker.tick.connect(self._on_tick)
        worker.done.connect(self._done)
        worker.failed.connect(self._failed)
        self.hold_open_while(worker, self._stop)
        worker.start()

    def _on_tick(self, permille: int, status: str) -> None:
        self._progress.setValue(permille)
        self._status.setText(status)

    def _stop(self) -> None:
        """Cancel means *stop the run*, not "close and leave it going".

        `stop_check` was threaded all the way through `run_comparison`, and nothing ever called
        `request_stop`: closing the dialog left the worker measuring to the end and writing a
        comparison nobody had asked for. Polled, so it ends on the next unit boundary rather
        than at once -- the dialog says so and waits (`Dialog.hold_open_while`).
        """
        self._worker.request_stop()
        self._cancel.setEnabled(False)
        self._status.setText("Stopping — finishing the current recording…")

    def _done(self, comparison) -> None:
        self.let_go()
        self.comparison = comparison
        self.accept()

    def _failed(self, exc) -> None:
        from detail.api.comparison import ComparisonStopped

        self.let_go()
        self._worker = None
        if isinstance(exc, ComparisonStopped):
            # A cancellation the user asked for is not an error to report back at them.
            self.reject()
            return
        self._reset_controls()
        QMessageBox.critical(self, "New comparison", str(exc))

    def _reset_controls(self) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._cancel.setEnabled(True)
        self._cancel.setText("Cancel")
        self._progress.hide()
        self._status.setText("")


class ComparisonsPage(QWidget):
    """The experiment's saved comparisons; launch a new one; open one to read it."""

    def __init__(self, experiment, on_open, on_created=None):
        super().__init__()
        self._experiment = experiment
        self._on_open = on_open
        # A comparison made *here* has to reach the rail too. Both openers navigate; only
        # `on_created` also refreshes the tree, and this page used the plain one -- so the new
        # comparison opened and was missing from the rail beside it until the next full rebuild.
        self._on_created = on_created or on_open

        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Comparisons"))
        self._caption_box = QVBoxLayout()
        layout.addLayout(self._caption_box)
        self._caption: QLabel | None = None
        # On the left under the heading, where every other collection page puts its "New …"
        # button (Analysis, Processing); it sat alone at the right here.
        new_btn = QPushButton("New comparison")
        new_btn.clicked.connect(self._new_comparison)
        layout.addWidget(new_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self._ids: list[str] = []
        self._table = IndexTable(
            ["id", "from", "compared across", "one observation is", "groups"],
            empty_text=(
                "(no comparisons yet — New comparison contrasts a behavior between "
                "groups of your recordings)"
            ),
        )
        self._table.view.doubleClicked.connect(self._open_selected)
        layout.addWidget(self._table)
        layout.addStretch()
        self._reload()

    def _reload(self) -> None:
        comparisons = self._experiment.comparisons
        if self._caption is not None:
            self._caption.deleteLater()
        self._caption = _index_caption(
            f"{len(comparisons)} comparison(s)", rows=len(comparisons)
        )
        self._caption_box.addWidget(self._caption)
        self._ids = [c.id for c in comparisons]
        rows = []
        for comparison in comparisons:
            record = comparison.record()
            path = record.get("aggregate_to") or []
            rows.append(
                (
                    comparison.id,
                    record.get("source_run_id", ""),
                    " · ".join(record.get("group_by") or []),
                    "|".join(path) or "one per recording",
                    str(len(record.get("groups") or [])),
                )
            )
        self._table.set_rows(rows, ids=self._ids)

    def _open_selected(self, index) -> None:
        key = row_id(index)
        if key is not None:
            self._on_open(key)

    def _new_comparison(self) -> None:
        dialog = ComparisonLaunchDialog(self._experiment, self)
        if (
            dialog.exec() == QDialog.DialogCode.Accepted
            and dialog.comparison is not None
        ):
            self._reload()
            self._on_created(dialog.comparison.id)


def _measure_order(measure_key: tuple):
    """Sort key: registry order for the measure, then the rest of the key alphabetically."""
    from detail.analysis.measures import MEASURES

    ids = list(MEASURES)
    name = measure_key[0] if measure_key else ""
    index = ids.index(name) if name in ids else len(ids)
    return (index, tuple(str(p) for p in measure_key[1:]))


def _cell(summary, unit: str) -> str:
    """A group's value in its own unit, with its spread and n when it has them."""
    from detail.analysis import comparison_report

    if summary is None:
        return "—"
    text, shown_unit = comparison_report.shown_value(summary.mean, unit)
    # No spread at n=1: "no variation observed" and "variation not observable" are different
    # facts, and §24 says so explicitly.
    if summary.sd is not None:
        spread, _ = comparison_report.shown_value(summary.sd, unit)
        text = f"{text} ± {spread}"
    out = f"{text} {shown_unit}".strip() if shown_unit else text
    return f"{out}  (n={summary.n})"


def _change(item) -> str:
    """What changed, as a reader reads it — or what happened when a percent cannot say."""
    from detail.analysis import comparison_report

    if item is None:
        return ""
    return comparison_report._change_text(item)


class _FigureGrid(QScrollArea):
    """A grid of figure thumbnails that uses the width it is given.

    **Responsive, because a fixed thumbnail wastes a wide window.** The first version pinned
    three columns at 300px, so widening the app added dead space between the columns rather than
    bigger figures — on a wide monitor most of the tab was empty. Columns and thumbnail width are
    both derived from the viewport, so the grid fills what it has and a figure gets as large as
    the space allows.

    The figures are rendered at 300 dpi, so a thumbnail well past 300px is still *down*-scaling
    from the source and stays sharp; :data:`_MAX_THUMB` stops one growing absurdly on an
    ultrawide.
    """

    #: The width a thumbnail wants. Columns are chosen to get near it, then widened to fill —
    #: so a wider window gives *bigger* figures first and more of them second.
    _TARGET_THUMB = 420
    #: Never wider than this: one enormous panel is a worse read than three good ones.
    _MAX_THUMB = 560
    #: And never more columns than this. Filling a 4K monitor with eleven 260px panels is
    #: technically using the space and practically unreadable.
    _MAX_COLUMNS = 4
    #: Thumbnail widths are rounded to this, so a drag produces a handful of layouts instead of
    #: one per pixel. Measured on a 21-figure comparison: a 400px drag went through **69**
    #: distinct widths, each rebuilding every thumbnail from its PNG at 140ms a pass -- about ten
    #: seconds of decoding on the UI thread, while a comment right above `_relayout` claimed a
    #: drag did not rescale every pixmap. 40px is finer than the eye reads on a thumbnail.
    _WIDTH_STEP = 40

    def __init__(self, columns: int = 3, width: int = 300):
        super().__init__()
        self.setWidgetResizable(True)
        self._paths: list[Path] = []
        self._empty = ""
        self._laid_out: tuple[int, int] | None = None
        self.show_figures([])

    def show_figures(self, paths: list[Path], *, empty_text: str = "") -> int:
        self._paths = [Path(p) for p in paths if Path(p).is_file()]
        self._empty = empty_text
        self._laid_out = None
        self._relayout()
        return len(self._paths)

    def geometry_for(self, viewport_width: int) -> tuple[int, int]:
        """``(columns, thumbnail width)`` for a viewport this wide.

        Takes the width rather than reading it, so the arithmetic is testable without a shown
        window — a `resize()` on an unshown widget does not update the viewport.
        """
        available = max(viewport_width - 24, 1)
        columns = min(self._MAX_COLUMNS, max(1, available // self._TARGET_THUMB))
        width = min(self._MAX_THUMB, (available // columns) - 12)
        step = self._WIDTH_STEP
        return columns, max(120, (max(120, width) // step) * step)

    def _geometry(self) -> tuple[int, int]:
        return self.geometry_for(self.viewport().width())

    def _relayout(self) -> None:
        columns, width = self._geometry()
        if self._laid_out == (columns, width):
            return
        self._laid_out = (columns, width)
        inner = QWidget()
        grid = QGridLayout(inner)
        for index, path in enumerate(self._paths):
            grid.addWidget(_FigureThumb(path, width), index // columns, index % columns)
        if not self._paths and self._empty:
            label = QLabel(self._empty)
            label.setWordWrap(True)
            grid.addWidget(label, 0, 0)
        grid.setRowStretch(grid.rowCount(), 1)
        self.setWidget(inner)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # `_relayout` returns immediately unless the quantized geometry actually changed. That
        # guard was already here and did nothing: the width is derived from the viewport, so it
        # changed on every pixel of the drag. `_WIDTH_STEP` is what makes the claim true.
        self._relayout()


class _FigureThumb(QLabel):
    """One figure, opening full size on double-click.

    Handing the file to the OS rather than opening it in a window here: a comparison figure is a
    PDF/PNG a user wants beside a manuscript, not something to pan around in.

    This used to cite the GUI style guide's *"no zoom/pan in v1"* and the one other enlarge
    affordance in the app. Both are gone — the rule was retired when the shared zoomable
    `ImageView` landed, and that other affordance turned out to be unreachable and was deleted.
    The choice stands on its own reason now, which is the only reason it ever had: these are
    **plots**, and zoom on plots is a separate decision (`03-universal-viewer.md`, OQ-VIEW-7),
    deferred by the person who asked for it.
    """

    def __init__(self, path: Path, width: int):
        super().__init__()
        self._path = path
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self.setPixmap(
                pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
            )
        self.setToolTip(f"{path.name} — double-click to open")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def mouseDoubleClickEvent(self, _event) -> None:
        _open_local_path(self, self._path, what="file")


def _dark_figures_box() -> QCheckBox:
    """The one "dark figures" control, so the Figures tab and the export dialog ask alike."""
    box = QCheckBox("Dark background (for slides)")
    box.setToolTip(
        "Figures are drawn on white for print unless this is ticked. Dark is the same design on "
        "a dark surface — for a talk, not a paper. The choice is written into the manifest."
    )
    return box


class ComparisonExportDialog(Dialog):
    """Where to write a comparison's publication tree, and in which figure formats.

    Everything a comparison produces is written every time — the long-form `stats.csv`, the
    per-measure tables, the contrast grid as LaTeX/Markdown/CSV, the figures, `summary.md` and the
    manifest — so this asks *where* and *which image formats* rather than offering to leave parts
    out. It lists the contents instead, because "export" without saying what you get is the reason
    people go looking in the folder.

    A destination row with Browse, options, and OK relabelled.
    """

    FORMATS = ("pdf", "svg", "png")

    def __init__(self, comparison, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export comparison")
        self._comparison = comparison

        layout = self.body()
        layout.addWidget(
            _instructions(
                "Writes stats.csv (one row per observation), units.csv (one row per animal, "
                "a column per group), the per-measure tables, the contrast grid as LaTeX, "
                "Markdown and CSV, ranking.csv and significant.csv (the statistics behind "
                "every result that cleared), methods.txt, the figures, summary.md and a "
                "manifest with a hash of every file."
            )
        )
        form = QFormLayout()
        self._dest = QLineEdit(str(comparison.dir))
        self._dest.setFont(MONO)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        dest_row = QHBoxLayout()
        dest_row.addWidget(self._dest, 1)
        dest_row.addWidget(browse)
        form.addRow("Destination:", dest_row)
        layout.addLayout(form)

        self._formats = {}
        for name in self.FORMATS:
            box = QCheckBox(
                {
                    "pdf": "PDF (vector — for a manuscript)",
                    "svg": "SVG (vector — for editing)",
                    "png": "PNG (raster — for a slide, and what this page shows)",
                }[name]
            )
            # The default every surface shares (detail#76): the API's, the CLI's and the
            # catalogue's, read off one constant rather than chosen again here.
            box.setChecked(name in comparison_report.DEFAULT_FORMATS)
            self._formats[name] = box
            layout.addWidget(box)
        self._dark = _dark_figures_box()
        layout.addWidget(self._dark)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        self.set_buttons(self._buttons)

    def theme(self) -> str:
        """The figures' surface: light for print unless the box was ticked."""
        return "dark" if self._dark.isChecked() else "light"

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Export to", self._dest.text() or str(self._comparison.dir)
        )
        if chosen:
            self._dest.setText(chosen)

    def _on_accept(self) -> None:
        if not self._dest.text().strip():
            QMessageBox.warning(self, "Export comparison", "A destination is required.")
            return
        if not self.formats():
            QMessageBox.warning(
                self, "Export comparison", "Pick at least one figure format."
            )
            return
        self.accept()

    def formats(self) -> tuple[str, ...]:
        return tuple(name for name, box in self._formats.items() if box.isChecked())

    def destination(self) -> Path:
        return Path(self._dest.text().strip())


@dataclass(frozen=True)
class _Reading:
    """Everything the detail page's two tables and its figure choice are built from.

    One object because they were built from three: `_overview_tab`, `_group_table` and
    `_overview_paths` each re-read `stats.csv` and re-ran the permutation floor, so opening a
    comparison paid for the same 200-draw ranking three times. A ranking scales with the
    observation count, so on a large experiment that is seconds of frozen window.
    """

    groups: list[tuple]
    summaries: list
    ranking: object | None

    @property
    def contrasts(self) -> list:
        return list(self.ranking.contrasts) if self.ranking is not None else []


def _read_comparison(comparison_dir: Path, *, record: dict) -> _Reading:
    """Read `stats.csv` and rank it. Pure and Qt-free, so it runs in a worker thread.

    Through `comparison_report.rank_record` — the one record → ranking mapping the export and
    a live screen also use (§22), so this page cannot show a different floor from theirs.
    """
    import csv

    stats = Path(comparison_dir) / comparison_report.STATS_FILENAME
    if not stats.is_file():
        return _Reading(groups=[], summaries=[], ranking=None)
    with open(stats, encoding="utf-8-sig") as handle:
        obs = comparison_report._observations(list(csv.DictReader(handle)))
    # Through `coverage_by_group`, or every group reports `coverage == n` -- the exact reading
    # the column exists to refuse.
    summaries = compare.summarize(obs, coverage_by_group=compare.coverage_by_group(obs))
    groups = sorted({s.group for s in summaries})
    ancestors = comparison_report.read_hierarchy(comparison_dir)
    ranking = comparison_report.rank_record(  # off-thread: the worker body
        obs, record, ancestors
    )
    if record.get("source") == "nodes" and ranking is not None:
        # The shown set (§24 *The node source*): a Measures tab of a thousand nodes times four
        # measures is not a table anyone reads; the ranking carries the rest.
        labels = {
            item.measure_key[1]
            for item in comparison_report.shown_items(ranking, ancestors)
        }
        summaries = [s for s in summaries if s.measure_key[1] in labels]
    return _Reading(groups=groups, summaries=summaries, ranking=ranking)


def labels_from_screen_text(labelled: list[dict]) -> str:
    """The screen page's line for the labels written from it
    (``api.comparison.labels_from_screen``): *Labelled from this screen: rearing (#1375),
    grooming (#1402, inside kept cluster #1390)*."""
    parts = []
    for row in labelled:
        where = f"#{row['node']}"
        if row.get("relation") == "under":
            where += f", inside kept cluster #{row.get('kept', '?')}"
        elif row.get("relation") == "above":
            where += f", above kept cluster #{row.get('kept', '?')}"
        # Read looser than the screen's own floor when it was labelled (§22): said here, so
        # the page lists a lead as a lead.
        where += comparison_report.kept_at_text(row)
        parts.append(f"{row['name']} ({where})")
    return "Labelled from this screen: " + ", ".join(parts)


class ComparisonDetailPage(QWidget):
    """One saved comparison, read in three views.

    **Lead with the answer.** The first version put a nineteen-row table first and a column of
    twenty full-width images under it, so the question a comparison is opened to ask — what
    differs, and by how much — was something you assembled yourself by scrolling. A user called
    the page underwhelming and they were right: the content was correct and the shape made it
    work to read.

    Three tabs, and no more. **Overview** is the handful of largest changes with their figures;
    **Measures** is the full table, in stable registry order, for reading a specific number;
    **Figures** is everything, as a grid. Three because a long page is worse than a short one
    even now that the content area scrolls (``gui/scroll.py``): scrolling makes content
    *reachable*, it does not make it findable — and because `cluster_workspace` sets the
    precedent that a tab count is justified rather than grown.

    The header carries what makes the numbers readable *before* any of them: which groups, how
    many recordings and animals, and — when it applies — why no floor was computed. That last
    one is the difference between "nothing differed" and "this design cannot tell".
    """

    def __init__(
        self, experiment, comparison_id: str, on_deleted=None, on_created=None
    ):
        super().__init__()
        self._experiment = experiment
        self._comparison = experiment.get_comparison(comparison_id)
        self._on_deleted = on_deleted
        self._on_created = on_created
        record = self._comparison.record()
        self._record = record

        # A QTabWidget inside a top-justified page collapses to its minimum without this — the
        # lesson `cluster_workspace` records at its own construction. The policy is the whole of
        # it: `pages.add_body_widget` hands an expanding child the stretch. The 520px floor that
        # used to sit here was a guess at what the tabs need, and a guess in this position is a
        # licence for Qt to squeeze the contents below what they need and clip them silently —
        # `cluster_workspace` had the same line and its scatter rendered at 172px because of it.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        for line in self._header_lines(record):
            label = QLabel(line)
            label.setWordWrap(True)
            layout.addWidget(label)

        self._drift_note = QLabel("")
        self._drift_note.setWordWrap(True)
        self._drift_note.hide()
        set_label_color(self._drift_note, WARN)
        layout.addWidget(self._drift_note)
        #: For a screen: the labels written from it, as its run holds them now (§22).
        self._labels_note = QLabel("")
        self._labels_note.setWordWrap(True)
        self._labels_note.hide()
        layout.addWidget(self._labels_note)
        self._notes_box = NotesBox(experiment, "comparison", comparison_id)
        layout.addWidget(self._notes_box)

        self._status = QLabel("")
        self._overview_figures = _FigureGrid(columns=4, width=230)
        self._figures = _FigureGrid(columns=3, width=300)
        self._reading: _Reading | None = None
        self._pngs: list[Path] = []

        self._tabs = QTabWidget()
        self._tabs.addTab(self._overview_tab(), "Overview")
        self._tabs.addTab(self._measures_tab(), "Measures")
        self._tabs.addTab(self._figures_tab(), "Figures")
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._status)
        self._show_existing_figures()
        # Drawn when the Figures tab is *opened*, not when the page is. Opening a comparison
        # stays free — `stats.csv` already holds every number the other two tabs show, and
        # paying for matplotlib to render panels nobody asked to see is the cost this page was
        # built to avoid. Going to the tab is asking to see them.
        self._auto_rendered = False
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Off the UI thread: reading `stats.csv` and ranking it is seconds of work at this
        # project's cohort sizes, and `drift()` shells out to `git`. Both ran inline in this
        # constructor, so opening a comparison froze the window for as long as they took.
        from detail.api import comparison as comparison_api

        self._loading = run_busy(
            self,
            lambda: (
                _read_comparison(  # off-thread
                    self._comparison.dir,
                    record=record,
                ),
                self._comparison.drift(),  # off-thread
                comparison_api.labels_from_screen(  # off-thread: reads the run's state
                    experiment, comparison_id
                ),
            ),
            self._on_read,
            label="Reading this comparison…",
        )

        # The end of a comparison is where you decide what to do next, and the answer is usually
        # another one over the same source — so the form opens **pre-filled from this one**. The
        # button does not say so: a label explaining its own behaviour is a label doing the work
        # the behaviour should do, and the form makes it obvious the moment it opens.
        actions = QHBoxLayout()
        again = QPushButton("New comparison")
        again.clicked.connect(self._new_seeded)
        actions.addWidget(again)
        export = QPushButton("Export")
        export.clicked.connect(self._export_elsewhere)
        actions.addWidget(export)
        actions.addStretch(1)
        layout.addLayout(actions)

        if on_deleted is not None:
            layout.addWidget(_divider())
            delete = QPushButton("Delete comparison")
            delete.clicked.connect(self._delete)
            layout.addWidget(delete)

    # -- the three views ------------------------------------------------------- #
    def _header_lines(self, record: dict) -> list[str]:
        """What makes the numbers readable, before any of them.

        The group names first, because "treatment vs baseline" is what the reader came for
        and `Comparison cmp_03` is not. Then how much data stands behind it — recordings *and*
        distinct animals, which is the fact that explains most surprises, and which the cohort
        inference already works out for the launch form.
        """
        groups = record.get("groups") or []
        lines = [
            "  vs  ".join(groups) if len(groups) > 1 else (groups[0] if groups else ""),
        ]
        path = record.get("aggregate_to") or []
        bits = [
            f"from {record.get('source_run_id', '')}",
            f"{record.get('n_observations', 0)} observation(s) over {len(groups)} group(s)",
            f"one observation is {'|'.join(path) or 'one per recording'}",
        ]
        cohort_line = self._cohort_line(record)
        if cohort_line:
            bits.insert(1, cohort_line)
        lines.append("  ·  ".join(b for b in bits if b))
        if record.get("design"):
            # The resolved design, in the sentence the record carries (§24 *Designs*): a
            # ranking read without it is a floor from an unknown null.
            lines.append(
                f"design: {record['design']} — {record.get('design_reason', '')}"
            )
        # What the ranking scores by (§24 *The statistic*), from the record's resolved field.
        lines.append(comparison_report.statistic_note(record))
        if comparison_report.left_out_sentence(record):
            lines.append(
                "left out: "
                + comparison_report.left_out_sentence(
                    record, self._experiment.recording_names()
                )
            )
        if record.get("source") == "nodes":
            lines.append(
                f"every cluster of {record.get('source_run_id', '')} at or above "
                f"{record.get('candidate_floor_s')} s, named or not, ranked on share of time"
            )
        if record.get("source") == "locations":
            exported = record.get("location_export") or {}
            # Only a comparison made before thin-part removal was taken out says "removed".
            thin = " (thin parts removed)" if exported.get("remove_thin_parts") else ""
            lines.append(
                f"movement and body shape from location export "
                f"{record.get('source_run_id', '')}{thin}; each measure "
                "tested on its own"
            )
        # A label this comparison scores that a screen over the same column chose (§24): the
        # one line that keeps the effect from reading as an independent finding.
        lines.append(comparison_report.screened_sentence(record))
        return [line for line in lines if line]

    def _cohort_line(self, record: dict) -> str:
        """`2 recordings from 1 animal` — the fact that explains why a floor is missing."""
        from detail.analysis import cohort

        try:
            metadata = self._experiment.recording_metadata
        except (OSError, ValueError):
            return ""
        recordings = record.get("recordings") or sorted(
            (record.get("provenance") or {}).get("recordings") or {}
        )
        if not recordings:
            return ""
        subject = cohort.infer_subject_column(
            metadata, recordings, group_by=record.get("group_by") or []
        )
        if subject is None:
            return f"{len(recordings)} recording(s)"
        distinct = len(
            {cohort.cell(metadata.get(r) or {}, subject) for r in recordings} - {""}
        )
        return f"{len(recordings)} recording(s) from {distinct} {subject}"

    def _overview_tab(self) -> QWidget:
        """The largest changes, with their figures. **Sorted by result, deliberately** — unlike
        the Measures table, whose stable order is what lets a reader find the same row twice.
        A panel whose whole job is "what should I look at" is the one place ordering by outcome
        is the answer rather than a hazard."""
        widget = QWidget()
        box = QVBoxLayout(widget)
        self._overview_body = QVBoxLayout()
        box.addLayout(self._overview_body)
        box.addWidget(self._overview_figures, 1)
        self._overview_body.addWidget(QLabel("(reading…)"))
        return widget

    def _measures_tab(self) -> QWidget:
        widget = QWidget()
        box = QVBoxLayout(widget)
        self._measures_body = QVBoxLayout()
        box.addLayout(self._measures_body, 1)
        # Stretch factor **0**, and the body's is 1. The spacer then gets only what the table
        # will not take -- and the table is already capped at its own rows by
        # `fit_table_height`, so a short one still hugs them. At factor 1 the spacer won: a
        # 10-row reference table was laid out at 192px of the 321px it asked for, scrollbar and
        # all, with the rest of the page blank beneath it.
        box.addStretch(0)
        self._measures_body.addWidget(QLabel("(reading…)"))
        return widget

    def _on_read(self, payload) -> None:
        """Fill both tabs from the one reading, once it is back off the worker thread."""
        reading, drift, from_screen = payload
        self._reading = reading
        if drift:
            self._drift_note.setText(
                "Inputs changed since this ran:\n" + "\n".join(drift)
            )
            self._drift_note.show()
        if from_screen:
            self._labels_note.setText(labels_from_screen_text(from_screen))
            self._labels_note.show()
        _clear_layout(self._overview_body)
        _clear_layout(self._measures_body)
        for widget in self._overview_widgets(reading):
            self._overview_body.addWidget(widget)
        self._measures_body.addWidget(self._group_table(reading))
        # The Overview grid picks its figures from the ranking, so it could not be filled until
        # now; the Figures grid already holds everything.
        self._overview_figures.show_figures(self._overview_paths(self._pngs))

    def _overview_widgets(self, reading: _Reading) -> list[QWidget]:
        out: list[QWidget] = []
        ranking = reading.ranking
        if ranking is not None and ranking.no_floor_reason:
            note = QLabel(f"No permutation floor: {ranking.no_floor_reason}.")
            note.setWordWrap(True)
            set_label_color(note, WARN)
            out.append(note)
        if ranking is None:
            # Fewer than two groups: nothing to contrast. Three or more rank as the largest
            # contrast between any two of them (§24 *Designs*), so they no longer land here;
            # the Measures tab holds every group side by side either way.
            out.append(QLabel("(this comparison has nothing to contrast)"))
            return out
        if not reading.contrasts:
            out.append(QLabel("(this comparison contrasts no measures)"))
            return out

        out.append(_heading("Largest changes"))
        if ranking.floor is not None:
            floor_text = comparison_report.floor_sentence(ranking).replace("**", "")
            if comparison_report.statistic_sentence(ranking):
                floor_text += " " + comparison_report.statistic_sentence(ranking) + "."
            floor_note = QLabel(floor_text)
            floor_note.setWordWrap(True)
            out.append(floor_note)
        # The paragraph a Methods section takes, generated from the record (detail#76), so
        # what a paper says ran is what ran. Muted: it is there to copy, not to read first.
        methods = QLabel(comparison_report.methods_sentence(self._record, ranking))
        methods.setObjectName("methods")
        methods.setWordWrap(True)
        methods.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        set_label_color(methods, status_color.MUTED)
        out.append(methods)
        kept = ranking.kept_items()
        if ranking.kept is not None:
            out.append(
                QLabel(
                    f"{len(kept)} kept after nesting suppression, of "
                    f"{len(ranking.clearing())} clearing the floor."
                )
            )
        # The kept set first — the findings — then the ranking's head: over a thousand nodes
        # the top of the ranking is where to look, and the kept nodes are what it found.
        top = kept + [c for c in reading.contrasts if c not in kept][: self._TOP_N]
        headers, rows = comparison_report.ranking_rows(ranking, self._record, top)
        table = QTableWidget(len(rows), len(headers))
        table.setObjectName("overview_contrasts")
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, cells in enumerate(rows):
            for column, value in enumerate(cells):
                table.setItem(row, column, QTableWidgetItem(value))
        make_columns_resizable(table)
        give_slack_to(table, 0)
        # Enabled only now, after every cell is written — a `QTableWidget` resorts itself
        # on each `setItem` call while sorting is on, which would scramble this fill loop.
        # `enable_sorting`, not a bare `setSortingEnabled`, so this starts in the ranked
        # order just built rather than silently pre-sorted by column 0 (see its docstring).
        enable_sorting(table)
        # Fixed: this is the top `_TOP_N`, so a scrollbar on it means the panel is not showing
        # the rows it just promised. It never competes with the figure grid below.
        _fit_table_height(table, fixed=True)
        out.append(table)
        return out

    def _figures_tab(self) -> QWidget:
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.addWidget(self._figures, 1)
        # The figures draw themselves on open, so this is the re-render — after an edit to the
        # comparison, or to pick the formats up again. Named for what it does now rather than
        # for the step it used to gate.
        self._export_btn = QPushButton("Redraw figures")
        self._export_btn.clicked.connect(self._export)
        self._dark_figures = _dark_figures_box()
        row = QHBoxLayout()
        row.addWidget(self._export_btn)
        row.addWidget(self._dark_figures)
        row.addStretch(1)
        box.addLayout(row)
        return widget

    #: How many rows the Overview shows before it stops being an overview.
    _TOP_N = 6

    def _export_elsewhere(self) -> None:
        """Write the publication tree wherever the user wants it, in the formats they pick."""
        from detail.gui.busy import run_progress

        dialog = ComparisonExportDialog(self._comparison, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        destination, formats = dialog.destination(), dialog.formats()
        theme = dialog.theme()
        self._status.setText("Exporting…")
        run_progress(
            self,
            lambda emit: self._comparison.export(
                destination, formats=formats, theme=theme, progress=emit
            ),
            lambda manifest: self._exported_elsewhere(destination, manifest),
            label="Exporting…",
            # Not `_export_failed`: that one re-enables the *Figures* tab's Render button, which
            # this path never disabled -- so a failed export left the wrong widget touched and
            # said "Render figures" over an error about somewhere else entirely.
            on_error=self._elsewhere_failed,
        )

    def _exported_elsewhere(self, destination, manifest: dict) -> None:
        self._status.setText(f"Written to {destination}")
        # An export *into* the comparison's own directory rewrites the very files the grids are
        # showing -- and drops any the new render superseded, so a re-export in a different
        # format set leaves the grid pointing at files that are gone.
        if Path(destination).resolve() == Path(self._comparison.dir).resolve():
            self._show_existing_figures()

    def _new_seeded(self) -> None:
        dialog = ComparisonLaunchDialog(
            self._experiment, self, seed=self._comparison.record()
        )
        if (
            dialog.exec() == QDialog.DialogCode.Accepted
            and dialog.comparison is not None
            and self._on_created is not None
        ):
            self._on_created(dialog.comparison.id)

    def _group_table(self, reading: _Reading) -> QWidget:
        """One row per measure, with a column per group and what changed — from `stats.csv`,
        no export and no re-measure.

        **Wide, not long.** The first version rendered one row per (measure, group): 38 rows for a
        two-behavior query, and the reader did the subtraction across pairs of them. The question
        a comparison is opened to answer is what differs and by how much, so the contrast is a
        column rather than an exercise.

        **Registry order, not size of change.** This table is a reference: the same measure sits
        in the same row in every comparison, so a reader who has learned where to look does not
        re-locate it each time. The *ranking* on Overview is an ordering by definition, and it
        ranks on ``|fold change|`` — the only scale that means anything across measures that do
        not share a unit. Both render in the measure's own unit, not the raw nanoseconds this
        table used to print (`4956702127.6595745`).
        """
        if not reading.summaries:
            return QLabel("(this comparison wrote no stats.csv)")
        groups = reading.groups
        by_measure: dict[tuple, dict[tuple, object]] = {}
        for summary in reading.summaries:
            by_measure.setdefault(summary.measure_key, {})[summary.group] = summary

        ranking = reading.ranking
        # Stable order, not "biggest change first". A row that moves between runs cannot be
        # found twice, and a reader who has to re-locate every measure each time has lost the
        # thing a table is for. The ranking *figure* is an ordering by definition; the table is
        # a reference. Registry order, so measures appear the way they are documented.
        keys = sorted(by_measure, key=_measure_order)

        headers = ["measure", *("|".join(g) for g in groups)]
        if ranking is not None and ranking.profile:
            # Each measure's change is between its own pair (§24 *Designs*), so the cell
            # names it rather than the header claiming one direction for every row.
            headers.append("change (most different pair)")
        elif ranking is not None:
            headers.append(f"change ({' → '.join('|'.join(g) for g in groups)})")
        table = QTableWidget(len(keys), len(headers))
        # Named so a test asks for the table it means. Two tables sit on this page, `findChildren`
        # returns them in an order neither one chose, and an assertion written against index 0
        # silently moved to the other one -- which is how the Overview's mislabelled columns went
        # on passing a test whose docstring said it was checking them.
        table.setObjectName("group_measures")
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        contrasts = (
            {c.measure_key: c for c in ranking.contrasts} if ranking is not None else {}
        )
        for row, key in enumerate(keys):
            unit = comparison_report.measure_unit(key)
            cells = [
                comparison_report.measure_title(
                    key, comparison_report.remainders_of(self._record)
                )
            ]
            for group in groups:
                cells.append(_cell(by_measure[key].get(group), unit))
            if ranking is not None:
                item = contrasts.get(key)
                change = _change(item)
                if ranking.profile and item is not None and item.pair is not None:
                    change = f"{change} ({item.pair[0]} → {item.pair[1]})"
                cells.append(change)
            for column, value in enumerate(cells):
                table.setItem(row, column, QTableWidgetItem(value))
        make_columns_resizable(table)
        # The measure name is the column that identifies the row, so it takes the slack --
        # stretching the last one instead leaves the change column half the table wide.
        give_slack_to(table, 0)
        # Deliberately not sortable: the docstring above is the reason — registry order is
        # the point of this table (a fixed position a reader learns), not the change ranking
        # a click-to-sort would invite. `_overview_widgets`'s top-N table is the ranked view.
        _fit_table_height(table)
        return table

    # -- figures --------------------------------------------------------------- #
    def _on_tab_changed(self, index: int) -> None:
        """Draw the figures the first time somebody opens the tab that shows them.

        A comparison that has never been exported opened on an empty grid and a button, so the
        figures — the reason the tab exists — were one click away from every reader, every time.
        Drawing them is a pure re-render of `stats.csv` (§18.5 L1: no recompute), so doing it
        unasked cannot change a result; it goes through `run_progress` like the button does,
        off the UI thread, because matplotlib is not fast.

        Once. A render that fails re-enables the button and leaves the flag set, so a broken
        export does not redraw itself every time the tab is touched.
        """
        if self._auto_rendered or self._tabs.tabText(index) != "Figures":
            return
        self._auto_rendered = True
        if not self._pngs:
            self._export()

    def _show_existing_figures(self) -> None:
        pngs = sorted((self._comparison.dir / "figures").glob("*.png"))
        if pngs:
            self._render(pngs)
            self._status.setText(
                "Showing the figures already written beside this comparison."
            )

    def _export(self) -> None:
        from detail.gui.busy import run_progress

        theme = "dark" if self._dark_figures.isChecked() else "light"
        self._export_btn.setEnabled(False)
        run_progress(
            self,
            # The same formats an export writes, so a redraw beside the comparison leaves
            # the tree a manuscript takes (PDF) and an editor opens (SVG), not only the PNG
            # this page shows.
            lambda emit: self._comparison.export(theme=theme, progress=emit),
            self._exported,
            label="Rendering figures…",
            on_error=self._export_failed,
        )

    def _exported(self, manifest: dict) -> None:
        self._export_btn.setEnabled(True)
        out = self._comparison.dir
        self._status.setText(f"Written to {out}")
        # A comparison writes one figure per measure, named after it, plus the difference
        # ranking only when there are exactly two groups — so read what was actually written
        # rather than asking for a fixed list of names.
        self._render(
            [
                out / name
                for name in sorted(manifest.get("artifacts") or {})
                if name.startswith("figures/") and name.endswith(".png")
            ]
        )

    def _render(self, pngs) -> None:
        """Fill both grids: everything in Figures, and the top rows' own figures in Overview."""
        paths = [Path(p) for p in pngs if Path(p).is_file()]
        self._figures.show_figures(
            paths,
            empty_text=(
                "No figures yet — “Redraw figures” draws one panel per measurement and "
                "writes them beside this comparison."
            ),
        )
        self._pngs = paths
        # Only once the reading is back: the Overview grid shows the *ranked* rows' figures, so
        # before then there is nothing to choose with. `_on_read` fills it.
        if self._reading is not None:
            self._overview_figures.show_figures(self._overview_paths(paths))

    def _overview_paths(self, paths: list[Path]) -> list[Path]:
        """The figures belonging to the rows the Overview shows, in the same order.

        A stem names its measurement, so the mapping is by name rather than by a second registry
        — which is also why the panels figure and the ranking come first: they are the panels
        about the whole comparison rather than one measure of it.
        """
        contrasts = self._reading.contrasts if self._reading is not None else []
        wanted = [
            comparison_report._measure_stem(
                comparison_report._measurement(c.measure_key)
            )
            for c in contrasts[: self._TOP_N]
        ]
        by_stem = {p.stem: p for p in paths}
        chosen = [by_stem[stem] for stem in wanted if stem in by_stem]
        whole = [
            by_stem[stem]
            for stem in (comparison_report.PANELS_STEM, "difference_ranking")
            if stem in by_stem
        ]
        return whole + chosen

    def _export_failed(self, exc) -> None:
        self._export_btn.setEnabled(True)
        self._status.setText("")
        QMessageBox.critical(self, "Redraw figures", str(exc))

    def _elsewhere_failed(self, exc) -> None:
        self._status.setText("")
        QMessageBox.critical(self, "Export comparison", str(exc))

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete comparison",
            "Delete this comparison and everything it wrote? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_comparison(self._comparison.id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))
            return
        self._on_deleted()


def _clear_layout(box) -> None:
    """Empty a layout, deleting the widgets it held."""
    while box.count():
        item = box.takeAt(0)
        discard(item.widget())


def _number(value) -> str:
    """A summary number as a person reads it, blank when there is none."""
    return "" if value is None else f"{value:.4g}"
