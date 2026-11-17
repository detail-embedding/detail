"""Launch-run dialog (docs/gui_design.md §4.13).

Generic over modules: pick a module, select eligible recordings (its ``upstream``
must be satisfied), fill the module's own config form, and launch via
``Experiment.run_processing``. The dialog holds the returned ``AttachedRun`` in
``run_handle`` for the shell to show live progress.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core.protocols import ModelNotReadyError
from detail.executor.runners import (
    ANALYSIS_MODULES,
    PROCESSING_MODULES,
    import_runner_modules,
)
from detail.gui import modules as _gui_modules  # noqa: F401 — registers UIs
from detail.gui.dialog import Dialog
from detail.gui.forms import AdvancedSection
from detail.gui.modules import registry
from detail.gui.start_from import (
    StartFromRow,
    analysis_runs_for,
    behavior_names,
    processing_runs_for,
    unrestored_settings,
)
from detail.gui.status_color import WARN, set_label_color
from detail.gui.modules.registry import (
    ANALYSIS_GUI_MODULES,
    GUI_MODULES,
    ConfigWidget,
)
from detail.gui.selectable import make_text_selectable
from detail.gui.sizing import fit_list_height
from detail.gui.workers import BackgroundCall, StageProgressWorker
from detail.storage import read_cache


def _resolve_recordings_in_background(dialog, selected: list, gen: int) -> None:
    """Resolve a config form's recording selection off the UI thread (a heavy config
    reads disk per recording — the select-all lag) then apply on the UI thread. A
    generation guard drops a result whose selection has since changed; the in-flight
    QThread is parked on the dialog so it isn't garbage-collected mid-run."""
    config = dialog._config
    call = BackgroundCall(lambda: config.resolve_recordings(selected), receiver=dialog)
    dialog._resolve_calls.add(call)

    def _finish() -> None:
        dialog._resolve_calls.discard(call)

    def _done(resolved) -> None:
        if gen == dialog._resolve_gen and dialog._config is config:
            _apply_to_config(dialog, selected, resolved)
        _finish()

    def _failed(_exc) -> None:
        _finish()  # a resolution failure leaves the form as-is (no crash)

    call.done.connect(_done)
    call.failed.connect(_failed)
    call.start()


def _push_config_recordings(dialog, sync: bool = False) -> None:
    """Push the dialog's checked recordings into its config form. A form that declares
    heavy per-recording resolution (overrides ``resolve_recordings``) is resolved off the
    UI thread; a cheap form (or an empty selection, or ``sync``) applies inline. Every
    call bumps the resolution generation so a stale off-thread result is dropped."""
    selected = [
        dialog._all_recordings[i]
        for i in range(dialog._recordings.count())
        if dialog._recordings.item(i).checkState() == Qt.CheckState.Checked
    ]
    dialog._resolve_gen += 1
    heavy = (
        type(dialog._config).resolve_recordings is not ConfigWidget.resolve_recordings
    )
    if sync or not selected or not heavy:
        _apply_to_config(dialog, selected, dialog._config.resolve_recordings(selected))
        return
    _resolve_recordings_in_background(dialog, selected, dialog._resolve_gen)


def _apply_to_config(dialog, selected: list, resolved) -> None:
    """Install a resolved selection in the dialog's form, noting which recordings it was.

    The form's coverage (``covered_recording_ids``) answers for these recordings and no
    others, so the dialog keeps the set to know which rows the answer is about. Noted
    first: applying emits ``coverage_changed``, and the rows are re-marked from inside it.
    """
    dialog._resolved_ids = {recording.id for recording in selected}
    dialog._config.apply_recordings(selected, resolved)


#: What a close request during a launch says, once it has asked the launch to stop.
STOPPING_LAUNCH = (
    "Stopping — if the run has already been started, it opens instead of this window."
)


def _set_form_busy(dialog, busy: bool) -> None:
    """Freeze a launch dialog's form while it validates + spawns off the UI thread,
    showing a status line so a multi-second validation reads as work, not a hang.
    Shared by both launch dialogs (identical field set).

    Cancel stays live and reads **Stop**: disabling it left Esc as the only way out, and Esc
    closed the dialog as cancelled while the worker went on to start the run (#35).
    """
    for widget in (
        dialog._ok_button,
        dialog._module_combo,
        dialog._recordings,
        dialog._select_all,
        dialog._clear,
    ):
        widget.setEnabled(not busy)
    dialog._cancel_button.setEnabled(True)
    dialog._cancel_button.setText("Stop" if busy else "Cancel")
    if dialog._config is not None:
        dialog._config.setEnabled(not busy)
    dialog._status.setText(
        f"Validating {len(dialog._pending_recordings)} recording(s)…" if busy else ""
    )


def _stop_launch(dialog, call: StageProgressWorker) -> None:
    """A close request while the launch runs: stop it at its next progress report.

    Every report ``run_processing`` makes comes before it reserves a run, so a stop that lands
    on one leaves no run behind; a launch that reports nothing more (``run_analysis`` reports
    nothing) starts its run, and the dialog then accepts and the run's own page opens. Either
    way the dialog closes on what actually happened, never on what was asked for.
    """
    call.request_cancel()
    dialog._cancel_button.setEnabled(False)
    dialog._status.setText(STOPPING_LAUNCH)


def _launch_in_background(dialog, work, error_title: str) -> None:
    """Run ``work(progress)`` (validates every recording + spawns the run) off the UI
    thread so the dialog never freezes — per-recording validation is O(recordings) disk
    I/O (§4.13, CLAUDE.md). Each ``(stage, fraction, detail)`` report replaces the status
    line, so a validation that takes minutes names the recording it is on. On success
    store the returned handle on ``dialog.run_handle`` and ``accept()``; on failure
    surface it and re-enable. Keeps the QThread referenced on the dialog so it isn't
    garbage-collected mid-run."""
    from detail.core.progress import OperationCancelled

    _set_form_busy(dialog, True)
    call = StageProgressWorker(work, receiver=dialog)

    def _tick(_permille: int, status: str) -> None:
        if (
            not call.cancel_requested()
        ):  # a report queued before the stop says nothing now
            dialog._status.setText(status)

    call.tick.connect(_tick)
    dialog._launch_call = call  # keep alive until it finishes
    dialog.hold_open_while(call, lambda: _stop_launch(dialog, call))

    def _done(handle) -> None:
        dialog.let_go()
        dialog.run_handle = handle
        dialog.accept()

    def _failed(exc) -> None:
        dialog.let_go()
        if isinstance(exc, OperationCancelled):
            dialog.reject()  # stopped before any run was reserved: nothing started
            return
        _set_form_busy(dialog, False)
        message = (
            str(exc)
            if isinstance(exc, (ValueError, ModelNotReadyError))
            else f"{type(exc).__name__}: {exc}"
        )
        QMessageBox.critical(dialog, error_title, message)

    call.done.connect(_done)
    call.failed.connect(_failed)
    call.start()


def _config_holder_layout(layout: QVBoxLayout) -> QVBoxLayout:
    """The layout a module's config form is hosted in.

    This used to wrap the form in its own `PageScroll`, and that was the shape behind
    *"windows open with no scroll bar, hiding content"*. It scrolled — the scrollbar was there
    and its range was right — but it was the **only** scroll in a dialog whose other seven
    top-level items could not scroll at all. Measured: a 117px viewport onto a 444px form, sat
    directly beside the Behaviors table, so the scrollbar read as that table's and the dialog
    looked like it ended.

    Now that `Dialog.body()` scrolls the whole dialog, a scroll here would be a same-axis
    scroller nested in another — which `gui/scroll.py` quotes Apple's HIG against, and which is
    exactly what made a porthole feel like a boundary. One scroll per window.

    The top-justify is `Dialog.set_buttons`'s, not this helper's: rows are still being added
    after this one (the status line), so a stretch here would push them to the bottom of the
    body with a gap above them.
    """
    holder = QVBoxLayout()
    holder.setContentsMargins(0, 0, 0, 0)
    layout.addLayout(holder)
    return holder


def _mark_config_baseline(config: ConfigWidget | None) -> None:
    """Take the form's opening state as the baseline its advanced sections call "changed".

    Called once the experiment and the recording selection have been pushed in — those fill
    combos, and a value the *form* chose is not a value the user changed. A seed applied
    after this does read as changed, which is the point: Refine opens with the section that
    holds it already showing.
    """
    if config is None:
        return
    for section in config.findChildren(AdvancedSection):
        section.snapshot()


def _start_from_note() -> QLabel:
    """The line under **Start from** that names what a pick could not restore; hidden until then."""
    note = QLabel("")
    note.setWordWrap(True)
    set_label_color(note, WARN)
    note.hide()
    return note


def _show_unrestored(dialog, module_id: str, picked: dict) -> None:
    """Name every setting of ``picked`` the freshly loaded form did not take (#35 finding 4).

    One note for every module, from the dialog, rather than one per form: a per-form note had
    reached two of the nine module forms.
    """
    lines = (
        unrestored_settings(
            module_id,
            picked,
            dialog._config.config(),
            noted_by_form=dialog._config.noted_by_form,
        )
        if picked
        else []
    )
    dialog._start_from_note.setText(
        "Not as in that run:\n" + "\n".join(lines) if lines else ""
    )
    dialog._start_from_note.setVisible(bool(lines))


def _discard_config(holder: QVBoxLayout, config: ConfigWidget | None) -> None:
    """Take the outgoing module's config form out of the holder, now.

    ``deleteLater`` alone only *schedules* the delete: until the event loop gets round to
    it the widget is still in the layout and still painted, so switching module stacked
    the new form under the old one (tracking's launcher showed detection's arguments).
    Unparenting is synchronous, so the swap is visible immediately; ``deleteLater`` then
    frees it safely even if we were called from one of its own signals."""
    if config is None:
        return
    holder.removeWidget(config)
    config.setParent(None)
    config.deleteLater()


# Per-row flag: does this recording have *any* complete run of the upstream stage?
_ELIGIBLE_ROLE = Qt.ItemDataRole.UserRole


def _eligible(recording, upstream: str) -> bool:
    """Whether a recording satisfies a module's ``requires`` entry.

    Most entries are **run kinds** (``detection`` / ``tracking`` / ``embedding``) —
    satisfied by a complete ``{kind}_runs`` record. Two are not runs: ``video`` (has a
    video part) and ``behavior`` (carries a behavior-role ground-truth stream, §3 — what
    the query eval scores against), so they resolve against recording state directly."""
    if upstream == "video":
        return bool(recording.video_parts)
    if upstream == "behavior":
        return recording.is_annotated
    runs = getattr(recording, f"{upstream}_runs", [])
    return any(r.get("status") == "complete" for r in runs)


def _covered_subset(config_widget, recordings: list[str], parent, title: str):
    """Narrow a launch to the recordings covered by the config's chosen upstream spec
    (§6.5), confirming the reduced set. Returns the covered ids, or None if the user
    cancels / none remain. ``covered_recording_ids() is None`` means all are covered."""
    covered = config_widget.covered_recording_ids()
    if covered is None:
        return recordings
    remaining = [sid for sid in recordings if sid in covered]
    dropped = [sid for sid in recordings if sid not in covered]
    if not remaining:
        QMessageBox.warning(
            parent,
            title,
            "None of the selected recordings has the chosen upstream run.",
        )
        return None
    if dropped:
        answer = QMessageBox.question(
            parent,
            title,
            f"{len(dropped)} selected recording(s) lack the chosen upstream run and "
            "will be skipped:\n"
            + ", ".join(dropped)
            + f"\n\nRun on the remaining {len(remaining)}?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
    return remaining


def _check_all_eligible(recording_list: QListWidget) -> None:
    """Check every *eligible* (enabled) recording in the list, leaving ineligible ones
    unchecked. Signals are blocked so the caller can emit a single config update after.
    """
    recording_list.blockSignals(True)
    for i in range(recording_list.count()):
        item = recording_list.item(i)
        if item.flags() & Qt.ItemFlag.ItemIsEnabled:
            item.setCheckState(Qt.CheckState.Checked)
    recording_list.blockSignals(False)


def _uncheck_all(recording_list: QListWidget) -> None:
    """Uncheck every recording in the list. Signals are blocked so the caller can emit a
    single config update after (mirror of ``_check_all_eligible``)."""
    recording_list.blockSignals(True)
    for i in range(recording_list.count()):
        recording_list.item(i).setCheckState(Qt.CheckState.Unchecked)
    recording_list.blockSignals(False)


# Launch-dropdown order: the pipeline itself, with the three core stages numbered.
# The keypoint stages (optional, after tracking and ahead of embedding, which may align on
# them — §26) stay in order but unnumbered; any future module falls
# back to alphabetical after these. An id here that is not registered is skipped, so the two
# keypoint ids take their place the day their modules land.
_PIPELINE_ORDER = [
    "detection",
    "tracking",
    "keypoint_detect",
    "keypoint_track",
    "embedding",
]
_STAGE_NUMBER = {"detection": 1, "tracking": 2, "embedding": 3}


def _ordered_module_ids(module_ids) -> list[str]:
    listed = [m for m in _PIPELINE_ORDER if m in module_ids]
    rest = sorted(m for m in module_ids if m not in _PIPELINE_ORDER)
    return listed + rest


# The analysis picker's running order: the ones people reach for, first, in the order a study
# actually goes. Everything else follows alphabetically. An id here that is not registered is
# simply skipped, so a module still on its own branch takes its place the day it lands rather
# than needing this list edited then. Ids are read off the modules
# (`gui/modules/analysis_*.py::module_id`), not guessed: a name that matches nothing would sort
# alphabetically forever and look like the ordering had simply been ignored.
#
# The two §22 discovery modules sit together and pose comes first, which is the order §22 gives
# them and the order they are useful in: what shapes does this animal make, then what does it do.
# Pose clustering is removed (`docs/removed_capabilities.md`) and expected back, so its id stays
# and is skipped until it registers again.
_ANALYSIS_ORDER = (
    "location_export",
    "pose_clustering",
    "temporal_clustering",
    "query",
)


def _ordered_analysis_ids(module_ids) -> list[str]:
    listed = [m for m in _ANALYSIS_ORDER if m in module_ids]
    return listed + sorted(m for m in module_ids if m not in _ANALYSIS_ORDER)


def _module_label(module_id: str, label: str) -> str:
    number = _STAGE_NUMBER.get(module_id)
    return f"{number}. {label}" if number else label


#: Record bookkeeping — not configuration, and a form must never see it as such.
_RECORD_KEYS = frozenset(
    {
        "params",
        "id",
        "module_id",
        "status",
        "created_at",
        "finished_at",
        "recording_ids",
    }
)


def seed_config(seed: dict) -> dict:
    """A run record flattened into the config dict shape ``ConfigWidget.config`` returns.

    A runner nests a run's knobs under ``params`` and *also* promotes a few keys to the
    record's top level, so a form reading one of the two levels sees a config with holes. Every
    ``load_config`` used to carry its own ``params.get(k) or record.get(k)`` fallback, which is
    why no two of them agreed on which keys survived a Refine — and why ``config()`` and
    ``load_config()`` could not be stated as inverses, let alone guarded as a round trip
    (``tests/gui/test_config_round_trip.py``).

    Flattened here, once, with ``params`` winning: ``params`` is the run's own config and a
    promoted key is a copy of it. A key promoted but *not* in ``params`` (``queries``) is kept.
    """
    params = seed.get("params") or {}
    promoted = {k: v for k, v in seed.items() if k not in _RECORD_KEYS}
    flat = {**promoted, **params}
    # A query-eval record lists what it scored as one dict per behavior; the form takes names.
    names = behavior_names(flat.get("behaviors"))
    if names is not None:
        flat["behaviors"] = names
    return flat


class LaunchRunDialog(Dialog):
    fits_content = True  # a settings form: show it all (Dialog.fits_content)

    def __init__(self, experiment, parent=None, seed=None):
        super().__init__(parent)
        self._experiment = experiment
        self.run_handle = None
        self.setWindowTitle("Launch a processing run")
        self.resize(560, 520)
        import_runner_modules(PIPELINE_RUNNERS)

        layout = self.body()
        form = QFormLayout()
        self._module_combo = QComboBox()
        for module_id in _ordered_module_ids(registry.surfaced(GUI_MODULES)):
            self._module_combo.addItem(
                _module_label(module_id, GUI_MODULES[module_id].label), module_id
            )
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)
        form.addRow("Module:", self._module_combo)
        self._start_from = StartFromRow()
        self._start_from.picked.connect(self._apply_start_from)
        form.addRow("Start from:", self._start_from)
        layout.addLayout(form)
        self._start_from_note = _start_from_note()
        layout.addWidget(self._start_from_note)

        layout.addWidget(QLabel("Recordings (check the ones to run over):"))
        self._recordings = QListWidget()
        self._recordings.itemChanged.connect(self._on_selection_changed)
        layout.addWidget(
            self._recordings
        )  # sized to its rows by `fit_list_height`, once they are known
        buttons = QHBoxLayout()
        self._select_all = QPushButton("Select all")
        self._select_all.clicked.connect(self._select_all_recordings)
        buttons.addWidget(self._select_all)
        self._clear = QPushButton("Clear")
        self._clear.clicked.connect(self._clear_all_recordings)
        buttons.addWidget(self._clear)
        buttons.addStretch()
        layout.addLayout(buttons)

        layout.addWidget(QLabel("Configuration:"))
        self._config_holder = _config_holder_layout(layout)
        self._config = None

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText("Run")
        self._cancel_button = box.button(QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        self._status = QLabel("")
        layout.addWidget(self._status)
        self.set_buttons(box)

        self._pending_recordings: list[str] = []
        self._resolve_gen = 0
        self._resolve_calls: set = set()
        self._resolved_ids: set[str] = set()  # what the form's coverage is about
        self._on_module_changed()
        if seed is not None:
            self._apply_seed(seed)

    def _apply_seed(self, seed: dict) -> None:
        """Pre-select a stage + recordings (a 'Run tracking →' next-step launch). The
        specific upstream run still comes from the config's picker."""
        idx = self._module_combo.findData(seed["module_id"])
        if idx >= 0:
            self._module_combo.setCurrentIndex(
                idx
            )  # rebuilds the recording list + config
        wanted = set(seed.get("recording_ids") or [])
        self._recordings.blockSignals(True)
        for row in range(self._recordings.count()):
            item = self._recordings.item(row)
            if self._recording_ids[row] in wanted and (
                item.flags() & Qt.ItemFlag.ItemIsEnabled
            ):
                item.setCheckState(Qt.CheckState.Checked)
        self._recordings.blockSignals(False)
        # A next-step seed may name the settings of the run it came from for a form's
        # *second* upstream picker (embedding's keypoint track run, §14). Handed over before
        # the recordings resolve, so the picker can pick it when its groups arrive.
        prefer = getattr(self._config, "prefer_upstream_spec", None)
        for upstream, spec in (seed.get("upstream_specs") or {}).items():
            if prefer is not None:
                prefer(upstream, spec)
        self._update_config_recordings()
        # Only when the seed carries a run's own knobs. This dialog is seeded two ways: a
        # *Refine* hands it the run being copied, and a "Run tracking →" next-step hands it the
        # **upstream** run's record with a *different* module_id — pushing detection's
        # parameters into tracking's form would be worse than not seeding at all. A next-step
        # seed is built from the record's recordings alone and has no `params`, which is the
        # difference to test on.
        if seed.get("params"):
            settings = seed_config(seed)
            self._config.load_config(settings)
            _show_unrestored(self, self._module_id(), settings)

    def _module_id(self) -> str:
        return self._module_combo.currentData()

    def _on_module_changed(self) -> None:
        module_id = self._module_id()
        upstream = PROCESSING_MODULES[module_id].upstream
        self._recordings.blockSignals(True)  # avoid per-item signals while rebuilding
        self._recordings.clear()
        # Pinned: the row wants a recording's name *and* its run lists, and each is its own
        # read of one `recording.yaml` (`read_cache.pinned`). Eligibility read one moment at a
        # time also made the list self-inconsistent — half of it could predate a run finishing.
        with read_cache.pinned():
            self._all_recordings = self._experiment.recordings
            self._recording_ids = [s.id for s in self._all_recordings]
            for recording in self._all_recordings:
                eligible = _eligible(recording, upstream)
                # The name is the handle (§11) — you pick recordings by what you called them.
                label = (
                    recording.name
                    if eligible
                    else f"{recording.name}   (needs {upstream})"
                )
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                # Eligibility is fixed for this module; coverage (below) varies with the
                # chosen upstream run. Kept apart so one never re-enables the other's rows.
                item.setData(_ELIGIBLE_ROLE, eligible)
                if not eligible:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self._recordings.addItem(item)
        self._recordings.blockSignals(False)
        fit_list_height(self._recordings)
        self._rebuild_config()
        self._start_from_note.hide()  # it was about the last module's pick
        self._start_from.set_module(
            self._experiment,
            module_id,
            processing_runs_for(self._experiment, module_id),
        )

    def _rebuild_config(self) -> None:
        """Replace the config form with a fresh one at the module's own defaults."""
        module_id = self._module_id()
        _discard_config(self._config_holder, self._config)
        self._config = GUI_MODULES[module_id].config_widget()
        self._config.coverage_changed.connect(self._mark_uncovered)
        self._config.on_experiment_set(self._experiment)
        self._config_holder.addWidget(self._config)
        self._update_config_recordings()
        _mark_config_baseline(self._config)
        # `Dialog.showEvent` covers a dialog's labels once, on show; this form is rebuilt
        # *while* the dialog is open — pick a different module and its rows are replaced —
        # and a show event does not fire again for that.
        make_text_selectable(self)

    def _apply_start_from(self, config: dict) -> None:
        """Set the form **to** the picked starting point — defaults, then its settings.

        A rebuild rather than an overlay, and the difference is the whole behaviour. A
        baseline names a handful of knobs; a run names another handful. Loading one over the
        other leaves a form that is part run, part baseline and wholly unchosen, and "Defaults"
        would leave everything exactly as it was. Rebuilding first makes every pick mean the
        same thing — *the form is now this* — with one notion of "reset" rather than two, and
        it reuses the path a module switch already takes.

        Applied *after* `_mark_config_baseline` (inside the rebuild), so a value the pick
        brought in reads as changed and the advanced section holding it opens itself.
        """
        self._rebuild_config()
        if config:
            self._config.load_config(config)
        _show_unrestored(self, self._module_id(), config)
        # A pick that lands inside Advanced settings must not be buried by it. `showEvent`
        # applies the same rule when a form is seeded before the dialog is shown; this is the
        # other moment it can become true.
        for section in self._config.findChildren(AdvancedSection):
            section.reveal_if_changed()

    def _on_selection_changed(self, _item) -> None:
        self._update_config_recordings()

    def _select_all_recordings(self) -> None:
        _check_all_eligible(self._recordings)
        self._update_config_recordings()

    def _clear_all_recordings(self) -> None:
        _uncheck_all(self._recordings)
        self._update_config_recordings()

    def _update_config_recordings(self, sync: bool = False) -> None:
        _push_config_recordings(self, sync)
        self._mark_uncovered()

    def _mark_uncovered(self) -> None:
        """Disable the checked recordings the chosen upstream run doesn't cover.

        Eligibility (``_eligible``) only asks whether a recording has *any* complete run
        of the upstream stage. Which run you picked narrows that further, and until now
        the difference surfaced at submit — as a prompt, or as a ValueError from
        ``run_processing`` for anyone launching another way. Marking it here makes the
        selection state honest while you are still choosing.

        Greys the row but leaves its check state alone, deliberately: the spec groups are
        computed from the *checked* recordings, so unchecking here would shrink the very
        set that decides coverage and the row could never come back (it ratcheted every
        recording out on the first pass). ``_selected_recordings`` skips disabled rows
        instead, so a greyed row cannot reach the run either way.

        Only a row the form resolved is judged. The coverage answers for the recordings the
        form was given, so a row outside them — unchecked, or checked and still resolving —
        has no answer yet. Reading that as "not covered" greyed every other recording the
        moment one was checked, and kept a row greyed for good once it was unchecked.
        """
        if self._config is None:
            return
        covered = self._config.covered_recording_ids()
        self._recordings.blockSignals(True)
        for row in range(self._recordings.count()):
            item = self._recordings.item(row)
            if not item.data(_ELIGIBLE_ROLE):
                continue  # ineligible for the stage — not ours to re-enable
            recording_id = self._recording_ids[row]
            name = self._all_recordings[row].name
            if (
                covered is None
                or recording_id in covered
                or recording_id not in self._resolved_ids
            ):
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                item.setText(name)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setText(f"{name}   (not in the selected run)")
        self._recordings.blockSignals(False)

    def _selected_recordings(self) -> list[str]:
        """Checked **and** available. A row greyed by ``_mark_uncovered`` keeps its tick
        (see there), so the check state alone would still submit a recording the chosen
        upstream run doesn't cover."""
        return [
            self._recording_ids[i]
            for i in range(self._recordings.count())
            if self._recordings.item(i).checkState() == Qt.CheckState.Checked
            and self._recordings.item(i).flags() & Qt.ItemFlag.ItemIsEnabled
        ]

    def _run(self) -> None:
        recordings = self._selected_recordings()
        if not recordings:
            QMessageBox.warning(self, "Launch run", "Select at least one recording.")
            return
        error = self._config.validation_error()
        if error:
            QMessageBox.warning(self, "Launch run", error)
            return
        recordings = self._narrow_to_covered(recordings)
        if recordings is None:
            return
        module_id = self._module_id()
        config = self._config.config()  # read widget state on the UI thread
        self._pending_recordings = recordings
        _launch_in_background(
            self,
            lambda progress: self._experiment.run_processing(
                module_id, recordings, config, progress=progress
            ),
            "Launch run",
        )

    def _narrow_to_covered(self, recordings: list[str]) -> list[str] | None:
        return _covered_subset(self._config, recordings, self, "Launch run")


class LaunchAnalysisDialog(Dialog):
    """Launch a experiment-level analysis run (§16): pick a module, select eligible
    recordings (its ``requires`` must be satisfied), fill the module's config form,
    and launch via ``Experiment.run_analysis``. Holds the returned ``AnalysisRun`` in
    ``run_handle`` for the shell to show live progress. Mirrors ``LaunchRunDialog``
    but over ``ANALYSIS_GUI_MODULES`` / ``ANALYSIS_MODULES``."""

    fits_content = True  # a settings form: show it all (Dialog.fits_content)

    def __init__(self, experiment, parent=None, seed=None):
        super().__init__(parent)
        self._experiment = experiment
        self.run_handle = None
        self.setWindowTitle("Launch an analysis run")
        self.resize(560, 520)
        import_runner_modules(PIPELINE_RUNNERS)

        layout = self.body()
        form = QFormLayout()
        self._module_combo = QComboBox()
        for module_id in _ordered_analysis_ids(registry.surfaced(ANALYSIS_GUI_MODULES)):
            self._module_combo.addItem(ANALYSIS_GUI_MODULES[module_id].label, module_id)
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)
        form.addRow("Module:", self._module_combo)
        self._start_from = StartFromRow()
        self._start_from.picked.connect(self._apply_start_from)
        form.addRow("Start from:", self._start_from)
        layout.addLayout(form)
        self._start_from_note = _start_from_note()
        layout.addWidget(self._start_from_note)

        layout.addWidget(QLabel("Recordings (check the ones to run over):"))
        self._recordings = QListWidget()
        self._recordings.itemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._recordings)
        buttons = QHBoxLayout()
        self._select_all = QPushButton("Select all")
        self._select_all.clicked.connect(self._select_all_recordings)
        buttons.addWidget(self._select_all)
        self._clear = QPushButton("Clear")
        self._clear.clicked.connect(self._clear_all_recordings)
        buttons.addWidget(self._clear)
        buttons.addStretch()
        layout.addLayout(buttons)

        layout.addWidget(QLabel("Configuration:"))
        self._config_holder = _config_holder_layout(layout)
        self._config = None

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText("Run")
        self._cancel_button = box.button(QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        self._status = QLabel("")
        layout.addWidget(self._status)
        self.set_buttons(box)

        self._pending_recordings: list[str] = []
        self._resolve_gen = 0
        self._resolve_calls: set = set()
        self._resolved_ids: set[str] = set()  # what the form's coverage is about
        self._on_module_changed()
        if seed:
            self._start_from_run(seed)

    def _start_from_run(self, seed: dict) -> None:
        """Open on one particular run's settings — the picker, pointed at a given run.

        This is what the *Refine* button used to do, and the only part of it worth keeping: it
        selects the module and loads the run's settings. It no longer pre-checks the run's
        recordings. That was the half of Refine that quietly disagreed with the rest of the
        design — recordings are a **selection**, chosen for the run you are about to make, and
        a picker that ticks boxes nobody asked it to tick is a picker people stop trusting. The
        item's own label says how many the source run used.
        """
        module_id = seed.get("module_id")
        index = self._module_combo.findData(module_id)
        if index >= 0:
            self._module_combo.setCurrentIndex(index)  # rebuilds the config form
        self._apply_start_from(seed_config(seed))

    def _module_id(self) -> str:
        return self._module_combo.currentData()

    def _requires(self, module_id: str) -> list[str]:
        return list(getattr(ANALYSIS_MODULES[module_id], "requires", []))

    def _on_module_changed(self) -> None:
        module_id = self._module_id()
        requires = self._requires(module_id)
        self._recordings.blockSignals(True)
        self._recordings.clear()
        with (
            read_cache.pinned()
        ):  # one read per recording, not one per `requires` entry
            self._all_recordings = self._experiment.recordings
            self._recording_ids = [s.id for s in self._all_recordings]
            for recording in self._all_recordings:
                eligible = all(_eligible(recording, r) for r in requires)
                need = " + ".join(requires)
                # The name is the handle (§11), as in `LaunchRunDialog` — this list alone kept
                # labelling rows `recording_001`.
                label = (
                    recording.name if eligible else f"{recording.name}   (needs {need})"
                )
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                if not eligible:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self._recordings.addItem(item)
        self._recordings.blockSignals(False)
        # Measured on the reported dialog: a `QListWidget` reports a constant 192px hint
        # whatever its row count — the same number as the table class — so one recording held
        # 192px of a 520px dialog and left the config area a **117px** viewport onto a 444px
        # form. Capping the list gives that back: 285px, most of the form visible instead of a
        # porthole beside a scrollbar that reads as the Behaviors table's.
        fit_list_height(self._recordings)
        self._rebuild_config()
        self._start_from_note.hide()  # it was about the last module's pick
        self._start_from.set_module(
            self._experiment, module_id, analysis_runs_for(self._experiment, module_id)
        )

    def _rebuild_config(self) -> None:
        """Replace the config form with a fresh one at the module's own defaults."""
        module_id = self._module_id()
        _discard_config(self._config_holder, self._config)
        self._config = ANALYSIS_GUI_MODULES[module_id].config_widget()
        self._config.on_experiment_set(self._experiment)
        self._config_holder.addWidget(self._config)
        self._update_config_recordings()
        _mark_config_baseline(self._config)
        # `Dialog.showEvent` covers a dialog's labels once, on show; this form is rebuilt
        # *while* the dialog is open — pick a different module and its rows are replaced —
        # and a show event does not fire again for that.
        make_text_selectable(self)

    def _apply_start_from(self, config: dict) -> None:
        """Set the form **to** the picked starting point — defaults, then its settings.

        A rebuild rather than an overlay: see `LaunchRunDialog._apply_start_from`. A pick means
        "the form is now this", so picking *Defaults* after a run has to undo the run, and
        picking a baseline after one must not leave the run's other values behind it.
        """
        self._rebuild_config()
        if config:
            self._config.load_config(config)
        _show_unrestored(self, self._module_id(), config)
        for section in self._config.findChildren(AdvancedSection):
            section.reveal_if_changed()

    def _on_selection_changed(self, _item) -> None:
        self._update_config_recordings()

    def _select_all_recordings(self) -> None:
        _check_all_eligible(self._recordings)
        self._update_config_recordings()

    def _clear_all_recordings(self) -> None:
        _uncheck_all(self._recordings)
        self._update_config_recordings()

    def _selected_recordings(self) -> list[str]:
        return [
            self._recording_ids[i]
            for i in range(self._recordings.count())
            if self._recordings.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _update_config_recordings(self, sync: bool = False) -> None:
        _push_config_recordings(self, sync)

    def _run(self) -> None:
        recordings = self._selected_recordings()
        if not recordings:
            QMessageBox.warning(
                self, "Launch analysis", "Select at least one recording."
            )
            return
        error = self._config.validation_error()
        if error:
            QMessageBox.warning(self, "Launch analysis", error)
            return
        recordings = _covered_subset(self._config, recordings, self, "Launch analysis")
        if recordings is None:
            return
        module_id = self._module_id()
        config = {"source_recording_ids": recordings, **self._config.config()}
        self._pending_recordings = recordings
        _launch_in_background(
            self,
            lambda _progress: self._experiment.run_analysis(module_id, config),
            "Launch analysis",
        )
