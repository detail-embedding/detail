"""*Estimate…* beside **Min contour area**: size the limit from the recordings' video.

Opened from the labeling forms (**Distill new model**'s detector page and **Label a dataset**),
it runs ``Experiment.estimate_min_contour_area`` off the UI thread (``busy.run_progress``, with
Cancel) over the recordings the form has selected, and shows:

- a row per recording: frames sampled, how many animals were seen at once, the animals' area
  (minimum, p1, median) and the leftover blobs' (median, p95), in pixels;
- the suggested value and how it was reached, or why there is none;
- what the estimate cannot see: it subtracts a median background rather than running the
  teacher, so a subject close in brightness to its background reads small and in pieces.

**Use this value** hands the suggestion back (:meth:`chosen_value`); the form writes it into
the field. Nothing is written to the experiment. The computation is
``detail.analysis.contour_area``; this window only shows it.
"""

from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from detail.gui.busy import run_progress
from detail.gui.dialog import Dialog
from detail.gui.sizing import set_available
from detail.gui.status_color import MUTED, set_label_color
from detail.gui.tables import SortCell, fit_table_height, fit_table_width, list_table

#: The estimate's limit, on screen in the window and in the docs (``docs/gui.md``).
LIMIT_NOTE = (
    "This measures moving blobs by subtracting the background, not the teacher's masks. A "
    "subject close in brightness to its background breaks into pieces, so its area here is a "
    "lower bound and the leftover blobs include pieces of it. Animals seen at once are counted "
    "and each is measured as an animal. Check the value with Test prompts before a long run."
)

_COLUMNS = (
    "Recording",
    "Frames",
    "Animals",
    "Animal min",
    "Animal p1",
    "Animal median",
    "Leftover median",
    "Leftover p95",
)


def _px(value: float | None) -> SortCell:
    if value is None:
        return SortCell("–", -1.0)
    return SortCell(f"{value:,.0f}", float(value))


class MinAreaEstimateDialog(Dialog):
    """Measure the selected recordings and suggest a *Min contour area* (module docstring).

    ``recording_ids`` are the form's selection; ``None`` means every recording with video, as
    the labeling run reads it.
    """

    fits_content = True

    def __init__(
        self,
        experiment,
        recording_ids: list[str] | None,
        *,
        parent=None,
        run_on_open: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Estimate min contour area")
        self._experiment = experiment
        self._recording_ids = list(recording_ids) if recording_ids else None
        self._result = None
        self._chosen: int | None = None
        self._running = False

        body = self.body()
        self._summary = QLabel(
            "Measure reads the recordings' video and suggests a value."
        )
        self._summary.setWordWrap(True)
        body.addWidget(self._summary)
        self._model = QStandardItemModel(0, len(_COLUMNS))
        self._model.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table = list_table()
        self._table.setModel(self._model)
        fit_table_height(self._table)
        body.addWidget(self._table)
        self._warnings = QLabel("")
        self._warnings.setWordWrap(True)
        self._warnings.hide()
        body.addWidget(self._warnings)
        note = QLabel(LIMIT_NOTE)
        note.setWordWrap(True)
        set_label_color(note, MUTED)
        body.addWidget(note)
        body.addStretch()

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._again = QPushButton("Measure")
        self._again.clicked.connect(self.run)
        box.addButton(self._again, QDialogButtonBox.ButtonRole.ActionRole)
        self._use = QPushButton("Use this value")
        self._use.clicked.connect(self._use_value)
        box.addButton(self._use, QDialogButtonBox.ButtonRole.AcceptRole)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._sync()
        if run_on_open:
            self.run()

    # -- public ---------------------------------------------------------------- #
    def chosen_value(self) -> int | None:
        """The value *Use this value* handed back, or None."""
        return self._chosen

    def estimate(self):
        """The last ``MinAreaEstimate`` shown, or None."""
        return self._result

    def run(self) -> None:
        """Measure off the UI thread (``Experiment.estimate_min_contour_area``)."""
        experiment, recording_ids = self._experiment, self._recording_ids
        self._running = True
        self._sync()

        def work(emit):
            return experiment.estimate_min_contour_area(  # off-thread
                recording_ids, progress=emit
            )

        run_progress(
            self,
            work,
            self.show_estimate,
            label="Measuring the recordings…",
            on_error=self._failed,
            on_cancel=self._cancelled,
            supersedes="estimate",
        )

    def show_estimate(self, result) -> None:
        """Fill the window from a ``MinAreaEstimate``."""
        self._running = False
        self._result = result
        self._model.removeRows(0, self._model.rowCount())
        for row in result.per_recording:
            name = SortCell(row.name, row.name)
            frames = SortCell(str(row.n_frames), row.n_frames)
            self._model.appendRow(
                [
                    name,
                    frames,
                    SortCell(str(row.n_animals), row.n_animals),
                    _px(row.animal_min),
                    _px(row.animal.get(1)),
                    _px(row.animal.get(50)),
                    _px(row.other.get(50)),
                    _px(row.other.get(95)),
                ]
            )
        self._table.resizeColumnsToContents()
        fit_table_height(self._table)
        fit_table_width(self._table)
        if result.suggestion is None:
            self._summary.setText(f"No suggestion. {result.reason}")
        else:
            self._summary.setText(f"Suggested: {result.reason}")
        self._warnings.setText("\n".join(result.warnings))
        self._warnings.setVisible(bool(result.warnings))
        self._sync()

    # -- internals ------------------------------------------------------------- #
    def _cancelled(self) -> None:
        self._running = False
        self._summary.setText("Cancelled. Measure runs it again.")
        self._sync()

    def _failed(self, exc: BaseException) -> None:
        self._running = False
        self._summary.setText(f"The estimate could not run: {exc}")
        self._sync()

    def _sync(self) -> None:
        busy = "Measuring the recordings…" if self._running else None
        set_available(self._again, busy is None, busy or "")
        if busy:
            reason = busy
        elif self._result is None:
            reason = "Nothing has been measured yet"
        elif self._result.suggestion is None:
            reason = "No value separates the animals from the leftover blobs"
        else:
            reason = None
        set_available(self._use, reason is None, reason or "")

    def _use_value(self) -> None:
        self._chosen = self._result.suggestion
        self.accept()


def min_area_row(spin: QSpinBox, on_estimate) -> QWidget:
    """The *Min contour area* field with *Estimate…* beside it.

    The row reads and announces through the spin box (``value`` / ``changed``), so the
    form's Advanced Settings counts an edit to it as it would the bare field.
    """
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(spin)
    button = QPushButton("Estimate…")
    button.setToolTip(
        "Measure the selected recordings' video and suggest a value for this field."
    )
    button.clicked.connect(on_estimate)
    layout.addWidget(button)
    layout.addStretch()
    row.value = spin.value
    row.changed = spin.valueChanged
    row.estimate_button = button
    return row


def open_estimate(form_widget, experiment, recording_ids, spin: QSpinBox) -> None:
    """Open the estimate for ``recording_ids`` and write a chosen value into ``spin``."""
    dialog = MinAreaEstimateDialog(
        experiment, recording_ids, parent=form_widget, run_on_open=True
    )
    if dialog.exec() and dialog.chosen_value() is not None:
        spin.setValue(dialog.chosen_value())
