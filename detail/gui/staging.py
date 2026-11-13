"""Watch, stop and resume the background video uploads a distill started (``executor/video_staging.py``).

**The question this answers.** ``Distill`` with *Upload videos in the background* copies a whole
video library to a cluster in a process that outlives the app, for as long as it takes (80
recordings was ~116 GB, about an hour). The only way to see it was ``detail remote
staging-status``, which a GUI user has no reason to know exists, so a healthy upload read as
nothing happening and a stuck one as the same.

**Reads off the UI thread, every time.** ``Experiment.video_staging`` asks the operating system
whether each upload's process is still alive, which shells out (PowerShell on Windows, ~hundreds
of ms), and ``stop_video_staging`` kills a process tree. Both are the known-slow seams
``tests/gui/test_ui_thread_blocking.py`` guards, so the poll and the stop run through
``BackgroundCall`` naming this dialog as their receiver: a result landing after the dialog closed
is dropped, not delivered to a deleted table.

**One ``state -> row`` mapping, here** (:func:`staging_rows`), and it is the one that is tested.
It carries the staging run's own numbers (files, GB, a measured ETA) rather than re-deriving a
fraction: the upload loop already weighs by bytes and times itself, and a second computation here
would be the second event-to-render mapping nobody tests.

**A table with no rows is a claim**, so it stays hidden until the first read lands; "(no
background uploads…)" shown while the read is in flight would tell someone with an upload running
that there is none.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QWidget,
)

from detail.core.progress import format_duration
from detail.gui.dialog import Dialog
from detail.gui.pages import IndexTable
from detail.gui.tables import row_id
from detail.gui.workers import BackgroundCall

COLUMNS = ["Target", "Status", "Files", "Sent", "Time left", "Detail"]

#: How often the open dialog re-reads. The upload's own state file is written about every half
#: second, so this is what keeps the numbers moving without being a read per frame.
POLL_MS = 2000

#: How often a run page's one-line status re-reads: slower than the dialog, because it is read in
#: passing beside a training curve and each read may ask the OS about a process.
LABEL_POLL_MS = 4000

EMPTY_TEXT = (
    "(no background uploads — tick Upload videos in the background in the Distill dialog, "
    "with a remote labeling or training target, to start one)"
)


def staging_rows(states: list[dict]) -> tuple[list[str], list[tuple]]:
    """The table's rows for ``Experiment.video_staging()``'s states, and each row's target.

    ``Sent`` counts the file in flight as well as the finished ones, so it keeps moving through a
    single multi-gigabyte file; ``Detail`` is the error when there is one, else the upload's own
    message.
    """
    ids, rows = [], []
    for state in states:
        eta = state.get("eta_s")
        ids.append(state["target"])
        rows.append(
            (
                state["target"],
                state["status"],
                f"{state['files_done']}/{state['files_total']}",
                _sent(state),
                format_duration(eta) if eta else "—",
                state.get("error") or state.get("message") or "",
            )
        )
    return ids, rows


def _sent(state: dict) -> str:
    """``12.5/116.0 GB``: finished files plus the one in flight, over everything to send."""
    sent = state["bytes_done"] + state.get("bytes_current", 0)
    return f"{sent / 1e9:.1f}/{state['bytes_total'] / 1e9:.1f} GB"


def staging_summary(target: str, state: dict | None) -> str:
    """One line for a distill run's page: where the videos are, said in the upload's own numbers.

    ``None`` is "no state recorded yet", which is the normal first seconds of a run (the
    coordinator starts the upload before it labels) and not an error. A stopped or failed upload
    says where to resume it, because this line is the only thing on the page that knows it needs
    that.
    """
    if state is None:
        return f"not started yet (to {target})"
    status = state["status"]
    if status == "running":
        eta = state.get("eta_s")
        return (
            f"uploading to {target} · {state['files_done']}/{state['files_total']} files · "
            f"{_sent(state)}" + (f" · ~{format_duration(eta)} left" if eta else "")
        )
    if status == "complete":
        return (
            f"on {target} · {state['files_total']} files, {_sent(state).split('/')[-1]}"
        )
    resume = " — resume it from Background uploads…"
    if status == "failed":
        return f"upload to {target} failed · {state.get('error') or 'no reason recorded'}{resume}"
    return (
        f"upload to {target} stopped at {state['files_done']}/{state['files_total']} "
        f"files{resume}"
    )


class UploadStatusLabel(QLabel):
    """The live line for one upload target, polled off the UI thread.

    Reads ``Experiment.video_staging`` (which asks the operating system whether the upload's
    process is alive, so never on the UI thread) every :data:`LABEL_POLL_MS` and shows
    :func:`staging_summary`. Polls for as long as it lives, not only while the upload runs: the
    person can resume a stopped one from the dialog and this line follows. The timer is the
    label's own child and the worker names the label as its receiver, so a page rebuild that
    deletes the label takes both with it.
    """

    def __init__(self, experiment, target: str, parent: QWidget | None = None) -> None:
        super().__init__("Reading…", parent)
        self.setWordWrap(True)
        self._experiment = experiment
        self._target = target
        self._reading = False
        self._timer = QTimer(self)
        self._timer.setInterval(LABEL_POLL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def refresh(self) -> None:
        if self._reading:
            return
        self._reading = True
        call = BackgroundCall(self._experiment.video_staging, receiver=self)
        call.done.connect(self._show_states)
        call.failed.connect(self._read_failed)
        call.start()

    def _show_states(self, states) -> None:
        self._reading = False
        state = next((s for s in states if s["target"] == self._target), None)
        self.setText(staging_summary(self._target, state))

    def _read_failed(self, exc) -> None:
        self._reading = False
        self.setText(f"could not read the upload: {exc}")


class BackgroundUploadsDialog(Dialog):
    """The background video uploads this experiment has started, live, with a way to stop one.

    Closing the dialog changes nothing: an upload belongs to its own process, not to this window.
    """

    def __init__(self, experiment, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Background uploads")
        self._experiment = experiment
        self._states: dict[str, dict] = {}
        self._reading = False

        layout = self.body()
        intro = QLabel(
            "Videos a distill is copying to a cluster while it labels and trains, so the "
            "detection run that follows finds them already there. A remote run that needs "
            "them waits for an upload still going rather than sending the same files twice."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self._caption = QLabel("Reading…")
        layout.addWidget(self._caption)
        self._table = IndexTable(COLUMNS, empty_text=EMPTY_TEXT, navigable=False)
        self._table.hide()
        self._table.view.selectionModel().selectionChanged.connect(
            lambda *_: self._sync_buttons()
        )
        # The view is as tall as its rows and the table widget around it is not, so left alone
        # the view sits in the middle of the extra height, a gap above it and a gap below.
        self._table.layout().setAlignment(self._table.view, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._table, 0, Qt.AlignmentFlag.AlignTop)
        # The table is as tall as its rows, so the slack goes below it, not between it and the
        # caption above.
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self._stop = QPushButton("Stop upload")
        self._stop.setToolTip(
            "Stop the selected upload. What already reached the cluster stays there, so a "
            "later run reuses it."
        )
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._stop_selected)
        buttons.addButton(self._stop, QDialogButtonBox.ButtonRole.ActionRole)
        self._resume = QPushButton("Resume upload")
        self._resume.setToolTip(
            "Start the selected upload again. Videos already on the cluster are skipped and a "
            "half-sent one continues, so only what was missing is sent."
        )
        self._resume.setEnabled(False)
        self._resume.clicked.connect(self._resume_selected)
        buttons.addButton(self._resume, QDialogButtonBox.ButtonRole.ActionRole)
        self.set_buttons(buttons)

        # Owned by the dialog, so it dies with it; started on show and stopped on hide so a
        # closed dialog is not polling.
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.refresh)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().showEvent(event)
        self._timer.start()
        self.refresh()

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        """Re-read the uploads, off the UI thread; a read still in flight is not stacked on."""
        if self._reading:
            return
        self._reading = True
        call = BackgroundCall(self._experiment.video_staging, receiver=self)
        call.done.connect(self._show_states)
        call.failed.connect(self._read_failed)
        call.start()

    def _show_states(self, states) -> None:
        self._reading = False
        selected = self._selected_target()
        self._states = {state["target"]: state for state in states}
        ids, rows = staging_rows(list(states))
        self._table.show()  # the read has happened, so an empty table now means something
        self._table.set_rows(rows, ids=ids)
        running = sum(1 for state in states if state["status"] == "running")
        self._caption.setText(
            f"{len(states)} upload(s), {running} running" if states else ""
        )
        self._reselect(selected)
        self._sync_buttons()

    def _read_failed(self, exc) -> None:
        self._reading = False
        self._caption.setText(f"Could not read the uploads: {exc}")

    def _selected_target(self) -> str | None:
        index = self._table.view.currentIndex()
        return row_id(index) if index.isValid() else None

    def _reselect(self, target: str | None) -> None:
        """``set_rows`` drops the selection, and a poll every two seconds would otherwise take
        it from someone who has just clicked a row to stop."""
        if target is None:
            return
        model = self._table.view.model()
        for row in range(model.rowCount()):
            if row_id(model.index(row, 0)) == target:
                self._table.view.selectRow(row)
                return

    def _sync_buttons(self) -> None:
        """Stop for a running upload, Resume for one that is not: the two are never both live."""
        state = self._states.get(self._selected_target() or "")
        running = state is not None and state["status"] == "running"
        self._stop.setEnabled(running)
        self._resume.setEnabled(state is not None and not running)

    def _stop_selected(self) -> None:
        target = self._selected_target()
        if target is not None:
            self._act(
                "stop the upload", lambda: self._experiment.stop_video_staging(target)
            )

    def _resume_selected(self) -> None:
        target = self._selected_target()
        if target is not None:
            self._act(
                "resume the upload",
                lambda: self._experiment.start_video_staging(target),
            )

    def _act(self, verb: str, fn) -> None:
        """Run a stop or a start off the UI thread (it checks, kills or spawns a process), then
        re-read. Both buttons go grey until it lands, so a double click is not two of them.
        """
        self._stop.setEnabled(False)
        self._resume.setEnabled(False)
        call = BackgroundCall(fn, receiver=self)
        call.done.connect(lambda _result: self.refresh())
        call.failed.connect(lambda exc: self._action_failed(verb, exc))
        call.start()

    def _action_failed(self, verb: str, exc) -> None:
        self._caption.setText(f"Could not {verb}: {exc}")
        self._sync_buttons()
