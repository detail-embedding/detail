"""*Tune tracking…* (§26 *Tuning a track method*): the keypoint tracking form's window for trying
``temporal_decode`` settings against an example set's labels.

The work is ``Experiment.tune_keypoint_tracking`` off the UI thread (``run_progress``, which
stops it if the window closes). The labelled-frames choice offers only sets that can score
(``Experiment.tune_example_sets``, read off the UI thread as the window opens): never the set the
detect run learned from, nor one labelled on another skeleton, and it names any it left out. The table lists every setting tried, detection alone and the
form's current settings first: PCK@0.1 on the *pick* half (what chose the recommendation) and on
the *check* half (labels the choice never saw — the figures to trust), with the check half's
paired and unpaired points beside them. The recommended row is selected; **Use selected
settings** hands any selected row's full config back to the form, and nothing is written.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from detail.gui.busy import run_busy, run_progress
from detail.gui.dialog import Dialog
from detail.gui.sizing import set_available
from detail.gui.tables import (
    give_slack_to,
    list_table,
    num_cell,
    refit_slack,
    stamp_row_id,
)
from detail.gui.tables import row_id as table_row_id
from detail.pipeline.keypoint_track_tune import CURRENT, DETECTION_ONLY

#: Pick / Check are PCK@0.1 over all points; the last two are the check half's left/right
#: points (the skeleton's pairs: paws, ears) and its midline (every point in no pair).
_COLUMNS = ("Setting", "Pick", "Check", "Check left/right", "Check midline")
_NO_SET = (
    "No example set to score with: it needs labelled frames, on the detect run's skeleton, "
    "and not to be the set the detect run learned from. Label a second example set, then "
    "come back."
)


def _pct(setting, half: str, name: str = "all") -> QStandardItem:
    figure = setting.figure(half, "group", name)
    value = None if figure is None else figure.get("value")
    if value is None:
        cell = QStandardItem("–")
        cell.setEditable(False)
        return cell
    return num_cell(value, f"{value:.1%}")


class KeypointTrackTuneDialog(Dialog):
    """Try tracking settings against labels; ``chosen_config`` is the config the user took."""

    def __init__(
        self,
        experiment,
        tune_detect_spec: dict,
        tune_base_config: dict,
        parent: QWidget | None = None,
        *,
        tracking_spec: dict | None = None,
        match: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tune tracking")
        self._experiment = experiment
        self._spec = tune_detect_spec
        self._base = dict(tune_base_config)
        # The track form's tracking run and match, so the poses reach the labelled subjects
        # the way the run would give them (§26 *Assignment*); None: the stage's defaults.
        self._tracking_spec = tracking_spec
        self._match = match
        self._result = None
        self.chosen_config: dict | None = None
        intro = QLabel(
            "Tracking is re-run on the videos that hold your labelled frames, once per "
            "setting (transition scale, smoothing, despike), and each is scored against the "
            "labels. Half of the labelled recordings pick the best setting; the other half "
            "check it, so trust the Check column. Nothing is written."
        )
        intro.setWordWrap(True)
        self.body().addWidget(intro)
        form = QFormLayout()
        self._sets = QComboBox()
        self._sets.setToolTip(
            "Labelled frames to score against. The example set the detect run learned from "
            "is left out (scoring on it would score the answer key), and so is a set "
            "labelled on another skeleton."
        )
        form.addRow("Labelled frames:", self._sets)
        self.body().addLayout(form)
        self._left_out = QLabel("")
        self._left_out.setWordWrap(True)
        self._left_out.setVisible(False)
        self.body().addWidget(self._left_out)
        # Which sets can score is read off the UI thread: telling the detect run's own set
        # from another hashes every set's images.
        self._sets_loaded = False
        self._run = QPushButton("Score settings")
        self._run.clicked.connect(self.score)
        self.body().addWidget(self._run)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self.body().addWidget(self._summary)
        self._model = QStandardItemModel(0, len(_COLUMNS))
        self._model.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table = list_table(stretch=False)
        self._table.setModel(self._model)
        # The window's primary content: it fills the window (the stretch factor), and the
        # setting's name takes the spare width rather than eliding.
        self.body().addWidget(self._table, 1)
        give_slack_to(self._table, 0)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._use = buttons.addButton(
            "Use selected settings", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._use.clicked.connect(self._take)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self._table.selectionModel().selectionChanged.connect(self._sync)
        self._sync()
        run_busy(
            self,
            lambda: experiment.tune_example_sets(tune_detect_spec),
            self._fill_sets,
            label="Reading the example sets…",
            on_error=self._sets_failed,
            supersedes="sets",
        )

    def _fill_sets(self, choices: list[dict]) -> None:
        """Offer the usable sets, the first selected; say which were left out and why."""
        self._sets_loaded = True
        self._sets.clear()
        for choice in choices:
            if choice["usable"]:
                self._sets.addItem(
                    f"{choice['name']} ({choice['id']}) · {choice['labelled']} labelled "
                    "frames",
                    choice["id"],
                )
        left = [c for c in choices if not c["usable"]]
        self._left_out.setText(
            "Left out: "
            + "; ".join(f"{c['name']} ({c['id']}), {c['reason']}" for c in left)
            + "."
            if left
            else ""
        )
        self._left_out.setVisible(bool(left))
        self._sync()

    def _sets_failed(self, exc: BaseException) -> None:
        self._sets_loaded = True
        self._summary.setText(f"Could not read the example sets: {exc}")
        self._sync()

    # -- state ------------------------------------------------------------------- #
    def _selected(self):
        if self._result is None:
            return None
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        name = table_row_id(rows[0])
        return next((s for s in self._result.settings if s.name == name), None)

    def _sync(self, *_args) -> None:
        has_sets = self._sets.count() > 0
        if not self._sets_loaded:
            set_available(self._run, False, "Reading the example sets…")
        else:
            set_available(self._run, has_sets, _NO_SET)
            if not has_sets and not self._summary.text():
                self._summary.setText(_NO_SET)
        chosen = self._selected()
        if chosen is None:
            reason = "Score the settings, then select a row"
        elif chosen.overrides is None:
            reason = "Detection alone is not a tracking setting; select another row"
        else:
            reason = ""
        set_available(self._use, not reason, reason)

    # -- scoring ----------------------------------------------------------------- #
    def score(self) -> None:
        set_id = self._sets.currentData()
        if set_id is None:
            return
        set_available(self._run, False, "Scoring…")
        experiment, spec, base = self._experiment, self._spec, self._base
        tracking, match = self._tracking_spec, self._match

        def work(emit):  # run_progress below runs this off the UI thread
            return experiment.tune_keypoint_tracking(  # off-thread
                set_id,
                keypoint_detect_spec=spec,
                tracking_spec=tracking,
                match=match,
                base_config=base,
                progress=emit,
            )

        run_progress(
            self,
            work,
            self._show,
            label="Scoring tracking settings…",
            on_error=self._failed,
            supersedes="tune",
            on_cancel=self._cancelled,
        )

    def _failed(self, exc: BaseException) -> None:
        self._summary.setText(f"Could not score: {exc}")
        self._sync()

    def _cancelled(self) -> None:
        self._summary.setText("Stopped.")
        self._sync()

    def _show(self, result) -> None:
        self._result = result
        self._model.removeRows(0, self._model.rowCount())
        best_row = None
        bold = QFont()
        bold.setBold(True)
        for setting in result.settings:
            name = QStandardItem(setting.name)
            name.setEditable(False)
            stamp_row_id(name, setting.name)
            if setting.name == result.best:
                name.setFont(bold)
                best_row = self._model.rowCount()
            self._model.appendRow(
                [
                    name,
                    _pct(setting, "pick"),
                    _pct(setting, "check"),
                    _pct(setting, "check", "paired"),
                    _pct(setting, "check", "unpaired"),
                ]
            )
        self._table.resizeColumnsToContents()
        refit_slack(self._table)
        if best_row is not None:
            self._table.selectRow(best_row)
        self._summary.setText(_summary(result))
        self._sync()

    def _take(self) -> None:
        chosen = self._selected()
        if chosen is None or chosen.overrides is None:
            return
        self.chosen_config = {**self._result.base_config, **chosen.overrides}
        self.accept()


def _summary(result) -> str:
    """One paragraph: what was scored, the recommendation, and what the check half says of it
    beside detection and the current settings."""
    cov = result.coverage

    half = "check" if result.checked else "pick"

    def check(name: str) -> str:
        figure = result.setting(name).figure(half, "group", "all")
        value = None if figure is None else figure.get("value")
        return "–" if value is None else f"{value:.1%}"

    lines = [
        f"{cov['scored']} labelled frames scored in {len(result.pick_recordings)} + "
        f"{len(result.check_recordings)} recordings"
        + (
            f"; {cov['excluded']} left out because the detect run learned from them"
            if cov["excluded"]
            else ""
        )
        + (
            f"; {cov['uncovered']} in recordings without this detect run"
            if cov["uncovered"]
            else ""
        )
        + ".",
        f"Recommended: {result.best}.",
    ]
    if result.checked:
        lines.append(
            f"On the check half it places {check(result.best)} of points within 0.1 body "
            f"of the label, against {check(DETECTION_ONLY)} for detection alone and "
            f"{check(CURRENT)} for the current settings."
        )
    else:
        lines.append(
            "Your labels are all in one recording, so there is no check half: these figures "
            "are the ones the setting was picked on, and flatter it. Label frames in more "
            "recordings for an honest check."
        )
    return " ".join(lines)
