"""Test a keypoint teacher (§12 *Distill keypoints*): *Test cut-off…* in the distill dialog
(:class:`KeypointTestDialog`), and *Test keypoints* on the keypoint detect form
(:class:`KeypointDetectTestDialog`, the same test laid out as detection's *Test prompts*).

The keypoints counterpart of detection's *Test prompts…*. It runs ``Experiment.test_keypoints``
off the UI thread (``busy.run_progress``), the same call ``detail keypoints test`` makes, on about
a dozen frames drawn **the way the distill would draw them**, and shows:

- each frame with its points: **filled** at or above the cut-off (taught as visible), **hollow**
  below it (placed, but not taught);
- a **Score cut-off** slider that redraws without running the teacher again, with how many of
  the placed points it keeps;
- for a method that learns from an example set, the **held-out check**: each example frame
  placed by the others and measured against the user's own click, in **pixels** with % of
  body size, above and below the cut-off, with counts (a median from fewer than 8 placements
  is withheld);
- **Another sample** (different frames, same kind of draw) and **Use this cut-off**, which
  hands the value back (:meth:`chosen_cutoff`).

Nothing is written: trying again is free.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)

from detail.core.visualize import draw_keypoints
from detail.gui.busy import run_progress
from detail.gui.confidence import ConfidenceRow
from detail.gui.dialog import Dialog
from detail.gui.full_frame import EnlargeableThumb, to_pixmap
from detail.gui.lifetime import discard
from detail.gui.sizing import lines, set_available
from detail.gui.spin import IntSpin
from detail.gui.status_color import MUTED, set_label_color

#: Cards per row, and how wide each picture is drawn.
_COLUMNS = 4
_CARD_WIDTH = 240


def _card_image(
    frame, skeleton, cutoff: float | None, *, full: bool = False
) -> np.ndarray:
    """The frame shrunk to a card (or at full size, for the window a double-click opens), then
    each track's points drawn: marks are sized to the image they go on, so drawing first and
    shrinking after loses them."""
    image = frame.image
    if not full and image.shape[1] > _CARD_WIDTH:
        height = max(1, round(image.shape[0] * _CARD_WIDTH / image.shape[1]))
        image = cv2.resize(image, (_CARD_WIDTH, height), interpolation=cv2.INTER_AREA)
    out = image
    # Points only, by the user's call (2026-10-09). The tracking box is still taught to the
    # student; it is just not drawn here.
    for _bbox, points, scores in frame.tracks.values():
        hollow = (
            None
            if cutoff is None
            else ~(np.nan_to_num(np.asarray(scores, float), nan=-np.inf) >= cutoff)
        )
        out = draw_keypoints(
            out, points, skeleton, radius=None if full else 4, hollow=hollow
        )
    return out


def _side(label: str, values: dict) -> str:
    if values["median_px"] is None:
        return f"{label}: {values['n']} placement(s), too few for a median"
    return (
        f"{label}: median {values['median_px']:.1f} px "
        f"({values['median_pct']:.1f}% of body size), "
        f"{values['near'] * 100:.0f}% of placements within 5% of body size, "
        f"n = {values['n']}"
    )


def held_out_text(report: dict) -> str:
    """The held-out check as lines a reader can scan: overall, then per point."""
    rows = [
        _side("At or above the cut-off", report["overall"]["above"]),
        _side("Below the cut-off", report["overall"]["below"]),
    ]
    for point, split in report["points"].items():
        rows.append(f"    {point}, " + _side("at or above", split["above"]))
    return "\n".join(rows)


def held_out_line(overall: dict) -> str:
    """The held-out check's overall result on one line: the median error above and below
    the cut-off, each with its count."""

    def short(label: str, values: dict) -> str:
        if values["median_px"] is None:
            return f"{label}, n = {values['n']} (too few for a median)"
        return (
            f"median {values['median_px']:.1f} px "
            f"({values['median_pct']:.1f}% of body size) {label}, n = {values['n']}"
        )

    return (
        short("at or above the cut-off", overall["above"])
        + "; "
        + short("below it", overall["below"])
    )


class KeypointTestDialog(Dialog):
    """Test a keypoint teacher on a representative sample (module docstring).

    ``teacher`` is the teacher keywords ``Experiment.test_keypoints`` takes (``detect_runs=``,
    or ``example_set=`` / ``tracking=`` / ``model_id=`` / ``model_variant=`` /
    ``label_compute=``); ``cutoff`` is where the slider opens. ``choose_cutoff=False`` drops
    *Use this cut-off* (then :meth:`chosen_cutoff` stays None). The detect form opens
    :class:`KeypointDetectTestDialog`, which overrides the layout hooks below.
    """

    fits_content = True

    def __init__(
        self,
        experiment,
        teacher: dict,
        *,
        cutoff: float | None = None,
        title: str = "Test cut-off",
        choose_cutoff: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(lines(78), lines(46))
        self._experiment = experiment
        self._teacher = dict(teacher)
        self._result = None
        self._seed = 0
        self._chosen: float | None = None
        self._running = False

        body = self.body()
        self._summary = QLabel(self._idle_text())
        self._summary.setWordWrap(True)
        self._build_top(body)
        self._legend = QLabel(self._legend_text())
        self._legend.setWordWrap(True)
        set_label_color(self._legend, MUTED)
        self._cards = QWidget()
        self._grid = QGridLayout(self._cards)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._cards)
        self._cutoff = ConfidenceRow(
            0.5 if cutoff is None else cutoff,
            heading="Score cut-off",
            help_text=self._cutoff_help(),
        )
        self._cutoff.threshold_changed.connect(lambda _v: self._redraw())
        if self.cutoff_above_frames:
            body.addWidget(self._cutoff)
            body.addWidget(self._legend)
            body.addWidget(scroll, 1)
        else:
            body.addWidget(self._legend)
            body.addWidget(scroll, 1)
            body.addWidget(self._cutoff)
        self._held = QLabel("")
        self._held.setWordWrap(True)
        self._held.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._build_held(body)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._another = QPushButton("Another sample")
        self._another.clicked.connect(self._another_sample)
        box.addButton(self._another, QDialogButtonBox.ButtonRole.ActionRole)
        self._use: QPushButton | None = None
        if choose_cutoff:
            self._use = QPushButton("Use this cut-off")
            self._use.clicked.connect(self._use_cutoff)
            box.addButton(self._use, QDialogButtonBox.ButtonRole.AcceptRole)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._sync()
        if self.run_on_open:
            self.run()

    # -- what a subclass changes (the detect form's window, below) ------------------ #
    #: Run the test as soon as the window opens (the distill's Test cut-off…).
    run_on_open = True
    #: The slider above the frames, as Test prompts has it, rather than below them.
    cutoff_above_frames = False

    def _idle_text(self) -> str:
        return "Testing the teacher on a sample of frames…"

    def _build_top(self, body) -> None:
        body.addWidget(self._summary)

    def _legend_text(self) -> str:
        return (
            "Filled: taught as visible. Hollow: placed, but below the cut-off, so not "
            "taught. Points are as the teacher placed them; nothing is smoothed."
        )

    def _cutoff_help(self) -> str:
        return "A point scored below the cut-off is taught as not visible."

    def _build_held(self, body) -> None:
        body.addWidget(self._held)

    def _teacher_for_run(self) -> dict:
        return dict(self._teacher)

    def _summary_text(self, result, scale: str, where: str) -> str:
        return (
            f"{result.method} on {len(result.frames)} of {result.requested} frames, drawn "
            f"the way the distill draws them (spread across recordings and over time). "
            f"Scores are {scale}. Ran on {where}."
        )

    def _caption(self, shown: int, placed: int) -> str:
        return f"taught visible: {shown} of {placed} points"

    def _show_held(self, report: dict | None, note: str | None) -> None:
        if note is not None:
            self._held.setText(f"Checked against your example clicks: {note}")
            return
        self._held.setText(
            "Checked against your example clicks (each example frame placed by the "
            "others):\n" + held_out_text(report)
        )

    # -- public ---------------------------------------------------------------- #
    def chosen_cutoff(self) -> float | None:
        """The cut-off *Use this cut-off* handed back, or None."""
        return self._chosen

    def cutoff(self) -> float:
        return float(self._cutoff.value())

    def teacher_test(self):
        """The last ``TeacherTest`` shown, or None. (Not ``result()``: that is QDialog's.)"""
        return self._result

    def run(self) -> None:
        """Test the teacher off the UI thread (``Experiment.test_keypoints``)."""
        experiment, teacher, seed = (
            self._experiment,
            self._teacher_for_run(),
            self._seed,
        )
        self._running = True
        self._sync()

        def work(emit):
            return experiment.test_keypoints(  # off-thread
                seed=seed, progress=emit, **teacher
            )

        run_progress(
            self,
            work,
            self._show,
            label=self._busy_label(),
            on_error=self._failed,
            on_cancel=self._cancelled,
        )

    # -- what came back ---------------------------------------------------------- #
    def _show(self, result) -> None:
        self._running = False
        self._result = result
        scale = (
            f"{result.score_meaning}, {result.score_low:g} to {result.score_high:g}"
            if result.score_meaning
            else "the method's own score"
        )
        where = result.device or "stored placements, no model run"
        self._summary.setText(self._summary_text(result, scale, where))
        self._redraw()

    def _cancelled(self) -> None:
        self._running = False
        self._summary.setText(self._cancelled_text())
        self._sync()

    def _failed(self, exc: BaseException) -> None:
        self._running = False
        self._summary.setText(f"The test could not run: {exc}")
        self._sync()

    def _redraw(self) -> None:
        result = self._result
        while self._grid.count():
            item = self._grid.takeAt(0)
            discard(
                item.widget()
            )  # not a bare deleteLater: the old card lingers on screen
        if result is None:
            self._sync()
            return
        cutoff = self.cutoff()
        skeleton = result.skeleton
        for i, frame in enumerate(result.frames):
            # A 240px card is too small to judge a paw by: double-click opens the frame full
            # size and zoomable, drawn at this cut-off (the cards are rebuilt on every move).
            card = EnlargeableThumb()
            card.setPixmap(to_pixmap(_card_image(frame, skeleton, cutoff)))
            card.set_render(
                lambda f=frame, c=cutoff: _card_image(f, skeleton, c, full=True),
                title=(
                    f"{frame.recording_id} {frame.part_id} frame {frame.frame_idx}, "
                    f"cut-off {cutoff:g}"
                ),
            )
            self._grid.addWidget(card, i // _COLUMNS, i % _COLUMNS)
        shown, placed = result.taught_visible(cutoff)
        self._cutoff.set_caption(self._caption(shown, placed))
        if result.held_out is None:
            self._show_held(None, "not available (this teacher has no example set).")
        elif not result.held_out:
            self._show_held(None, "needs at least two example frames.")
        else:
            from detail.pipeline.keypoint_preview import cutoff_report

            self._show_held(cutoff_report(result.held_out, cutoff), None)
        self._sync()

    def _busy_label(self) -> str:
        return "Testing the teacher…"

    def _cancelled_text(self) -> str:
        return "Cancelled. Another sample runs the test again."

    def _sync(self) -> None:
        reason = self._busy_label() if self._running else None
        set_available(self._another, reason is None, reason or "")
        if self._use is not None:
            no_result = reason or (None if self._result else "The test has not run yet")
            set_available(self._use, no_result is None, no_result or "")

    def _another_sample(self) -> None:
        self._seed += 1
        self.run()

    def _use_cutoff(self) -> None:
        self._chosen = round(self.cutoff(), 3)
        self.accept()


class KeypointDetectTestDialog(KeypointTestDialog):
    """*Test keypoints* on the keypoint detect form: the same test, laid out the way the
    detection form's *Test prompts…* is (settings and a Run button on top, the cut-off slider
    above the frames). A detect run stores raw points, so there is no *Use this cut-off*: the
    slider shows what a viewer or an export would hide at a cut-off.

    ``teacher`` is the form's patch-method keywords (``example_set`` / ``tracking`` /
    ``model_id`` / ``model_variant`` / ``label_compute`` / ``settings`` /
    ``every_n_frames``); ``recording_names`` maps each of its tracking's recordings to the
    name the *Recordings* choice shows.
    """

    run_on_open = False
    cutoff_above_frames = True

    def __init__(
        self,
        experiment,
        teacher: dict,
        recording_names: dict[str, str],
        *,
        cutoff: float | None = None,
        parent=None,
    ) -> None:
        self._recording_names = dict(recording_names)
        super().__init__(
            experiment,
            teacher,
            cutoff=cutoff,
            title="Keypoint test",
            choose_cutoff=False,
            parent=parent,
        )

    def _idle_text(self) -> str:
        return "Choose the frames, then Run keypoint test."

    def _busy_label(self) -> str:
        return "Testing keypoints…"

    def _cancelled_text(self) -> str:
        return "Cancelled. Run keypoint test runs it again."

    def _build_top(self, body) -> None:
        form = QFormLayout()
        self._recordings = QComboBox()
        self._recordings.addItem(f"All selected ({len(self._recording_names)})", None)
        for recording_id, name in self._recording_names.items():
            self._recordings.addItem(name, recording_id)
        self._recordings.setToolTip(
            "Test on frames spread across every selected recording, or on one."
        )
        form.addRow("Recordings:", self._recordings)
        self._n = IntSpin()  # how many frames to sample: this window's own knob
        self._n.setRange(1, 100)
        self._n.setValue(12)
        self._n.setToolTip("How many tracked frames to place points on.")
        form.addRow("Sample frames:", self._n)
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run keypoint test")
        self._run_btn.setToolTip(
            "Place this method's points on the sampled frames. Creates no run and writes "
            "nothing."
        )
        self._run_btn.clicked.connect(self.run)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._summary, 1)
        form.addRow("", run_row)
        body.addLayout(form)

    def _legend_text(self) -> str:
        return (
            "Filled: at or above the cut-off. Hollow: below it, so not visible in a "
            "viewer or export at this cut-off. Raw points, before tracking and smoothing."
        )

    def _cutoff_help(self) -> str:
        return (
            "A point scored below the cut-off is shown as not visible. The run keeps "
            "every point and its score; the cut-off is chosen where points are viewed or "
            "exported."
        )

    def _build_held(self, body) -> None:
        row = QHBoxLayout()
        row.addWidget(self._held, 1)
        self._per_point = QPushButton("Per point")
        self._per_point.setCheckable(True)
        self._per_point.setToolTip("Show the example set check for each point.")
        self._per_point.toggled.connect(lambda _on: self._redraw())
        set_available(self._per_point, False, "Run keypoint test first")
        row.addWidget(self._per_point, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(row)

    def _teacher_for_run(self) -> dict:
        teacher = dict(self._teacher)
        chosen = self._recordings.currentData()
        if chosen is not None:
            teacher["tracking"] = {chosen: teacher["tracking"][chosen]}
        teacher["n_frames"] = int(self._n.value())
        return teacher

    def _summary_text(self, result, scale: str, where: str) -> str:
        return (
            f"{result.method} on {len(result.frames)} of {result.requested} frames, "
            f"spread across the recordings and over time. Scores are {scale}. Ran on "
            f"{where}."
        )

    def _caption(self, shown: int, placed: int) -> str:
        return f"{shown} of {placed} points shown"

    def _show_held(self, report: dict | None, note: str | None) -> None:
        set_available(
            self._per_point, report is not None, "No per-point check for this sample"
        )
        if note is not None:
            self._held.setText(f"Example set check: {note}")
            return
        if self._per_point.isChecked():
            self._held.setText(
                "Example set check (each marked frame placed from the others, compared "
                "with your clicks):\n" + held_out_text(report)
            )
            return
        self._held.setText("Example set check: " + held_out_line(report["overall"]))

    def _sync(self) -> None:
        super()._sync()
        reason = self._busy_label() if self._running else None
        set_available(self._run_btn, reason is None, reason or "")
        if reason is None and self._result is None:
            set_available(self._another, False, "Run keypoint test first")
