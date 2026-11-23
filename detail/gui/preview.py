"""Synchronous inspection dialogs: prompt-test (§8) and render-preview (§14).

Each samples a few frames from one recording, runs the operation off the UI thread
(``save_as`` writes JPGs to a known dir), and shows them in a thumbnail grid.
Both need model/data that lives on a GPU box, so the success path is validated
there; the dialog, its worker, the grid, and the error path run anywhere.
"""

import json
import time
import traceback
from pathlib import Path

from detail.api import run_spec
from detail.core.dataclasses import Detection, Prompt, PromptTarget, TextCue
from detail.core.utils import decode_mask
from detail.core.visualize import assign_colors, draw_detections
from detail.gui.confidence import ConfidenceRow, showing_caption
from detail.gui.dialog import Dialog
from detail.gui.full_frame import EnlargeableThumb
from detail.gui.qt import (
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPixmap,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.run_preview import _to_pixmap
from detail.gui.spin import IntSpin
from detail.gui.workers import GuiWorker
from detail.pipeline._detection_worker import _encode_mask


def _save_name(prefix: str) -> str:
    return f"gui_{prefix}_{int(time.time() * 1000)}"


class _PreviewWorker(GuiWorker):
    """Runs a no-arg callable returning the output directory, off the UI thread."""

    finished_dir = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *, receiver):
        super().__init__(receiver=receiver)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except (
            Exception
        ) as exc:  # isolation boundary: surface any failure in the dialog
            traceback.print_exc()  # full traceback to the terminal for debugging
            self.emit_to_ui(self.failed, f"{type(exc).__name__}: {exc}")
            return
        self.emit_to_ui(self.finished_dir, result)


class _Swatch(QLabel):
    """A small filled square showing a prompt's overlay color."""

    def __init__(self, rgb):
        super().__init__()
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(*rgb))
        self.setPixmap(pixmap)


class _PromptHeader(QWidget):
    def __init__(self, text: str, rgb, total: int):
        super().__init__()
        self._text = text
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.addWidget(_Swatch(rgb))
        self._label = QLabel(f"{text}\n({total} total)")
        row.addWidget(self._label)
        row.addStretch()

    def set_total(self, total: int) -> None:
        """The column's total *at the current threshold* — it moves with the slider."""
        self._label.setText(f"{self._text}\n({total} total)")


class _MatrixCell(QWidget):
    """One overlay thumbnail plus its count, both redrawn when the threshold moves.

    220px on a 1920px recording is a 0.115 scale, so double-clicking the thumbnail opens the
    same overlay full size (`gui/full_frame.py`). The cell keeps the recipe rather than the
    picture, so the enlarged frame is drawn at whatever threshold the slider is on now.
    """

    _THUMB = 220

    def __init__(self):
        super().__init__()
        column = QVBoxLayout(self)
        column.setContentsMargins(2, 2, 2, 2)
        column.setSpacing(2)
        self._image = EnlargeableThumb("rendering…")
        self._image.setMinimumSize(self._THUMB, int(self._THUMB * 0.6))
        self._count = QLabel("")
        column.addWidget(self._image)
        column.addWidget(self._count)

    def show_overlay(self, rgb, count: int, *, render=None, title: str = "") -> None:
        self._image.setPixmap(
            _to_pixmap(rgb).scaled(
                self._THUMB,
                self._THUMB,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._image.set_render(render, title=title)
        self._count.setText(f"{count} det")


def _prompt_text(prompt_dict: dict) -> str:
    targets = prompt_dict.get("targets", [])
    if not targets:
        return "?"
    cue = targets[0].get("cue")
    return cue if isinstance(cue, str) else targets[0].get("class_name", "?")


class _CompactFrame:
    """One sampled frame's results with masks kept **run-length encoded**.

    The dialog has to hold every sampled frame to redraw it at a new threshold, and the
    run now keeps every proposal the model found. A full-resolution mask is ~2 MB, so
    holding them decoded across frames × prompts is gigabytes; encoded, a mask is
    kilobytes and decodes in the draw that needs it. This is why the dialog consumes
    results through ``on_result`` rather than taking the list.
    """

    def __init__(self, result):
        self.frame_idx = result.frame_idx
        self.frame_image = result.frame_image
        self.size = (result.frame_image.shape[0], result.frame_image.shape[1])
        self.rows: dict[str, list[tuple]] = {
            label: [
                (_encode_mask(d.mask), d.bbox, d.class_name, float(d.confidence))
                for d in det.detections()
            ]
            for label, det in result.results.items()
        }

    def detections(self, label: str, min_conf: float) -> list[Detection]:
        return [
            Detection(
                mask=decode_mask(rle, self.size),
                bbox=bbox,
                class_name=class_name,
                confidence=confidence,
                object_id=None,
            )
            for rle, bbox, class_name, confidence in self.rows.get(label, [])
            if confidence >= min_conf
        ]

    def count(self, label: str, min_conf: float) -> int:
        return sum(1 for row in self.rows.get(label, []) if row[3] >= min_conf)


class _PromptMatrix(QScrollArea):
    """Prompt-test results as a grid: columns = prompts (color swatch + total),
    rows = sampled frames, each cell the overlay thumbnail + its detection count.

    Renders from the run's detections rather than pre-baked JPGs, so the confidence
    threshold can be moved after the run — a written JPG is one threshold forever. This
    is the same shape the Quick segment comparison grid uses.
    """

    def __init__(self, frames: list, prompts: list, colors: dict, min_conf: float):
        super().__init__()
        self.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        self.setWidget(inner)
        self._frames = frames
        self._prompts = prompts
        self._colors = colors
        self._min_conf = min_conf
        self._cells: dict[tuple[int, str], _MatrixCell] = {}
        self._headers: dict[str, _PromptHeader] = {}
        if not frames:
            grid.addWidget(QLabel("(no results)"), 0, 0)
            return
        grid.addWidget(QLabel("frame"), 0, 0)
        for col, (label, text) in enumerate(prompts, start=1):
            header = _PromptHeader(text, colors.get(label, (200, 200, 200)), 0)
            grid.addWidget(header, 0, col)
            self._headers[label] = header
        for row, frame in enumerate(frames, start=1):
            grid.addWidget(QLabel(str(frame.frame_idx)), row, 0)
            for col, (label, _text) in enumerate(prompts, start=1):
                cell = _MatrixCell()
                grid.addWidget(cell, row, col)
                self._cells[(row - 1, label)] = cell
        self.set_min_conf(min_conf)

    def set_min_conf(self, value: float) -> None:
        """Redraw every cell at a new display threshold — no model, no decode."""
        self._min_conf = value
        for label, header in self._headers.items():
            header.set_total(sum(f.count(label, value) for f in self._frames))
        for i, frame in enumerate(self._frames):
            for label, _text in self._prompts:
                cell = self._cells.get((i, label))
                if cell is None:
                    continue
                dets = frame.detections(label, value)
                color = self._colors.get(label, (255, 255, 255))
                overlay = draw_detections(frame.frame_image, dets, color)
                # The *current* threshold, read when the cell is opened rather than captured
                # here: `set_min_conf` runs again on every slider move, so a cell always
                # carries a recipe for what it is showing.
                render = lambda f=frame, lb=label, c=color: draw_detections(
                    f.frame_image, f.detections(lb, self._min_conf), c
                )
                cell.show_overlay(
                    overlay,
                    len(dets),
                    render=render,
                    title=f"frame {frame.frame_idx} · {self._prompt_text(label)}",
                )

    def _prompt_text(self, label: str) -> str:
        return next((text for lb, text in self._prompts if lb == label), label)

    def counts(self, min_conf: float) -> tuple[int, int]:
        """``(kept at min_conf, total stored)`` across every frame and prompt."""
        kept = total = 0
        for frame in self._frames:
            for label, _text in self._prompts:
                kept += frame.count(label, min_conf)
                total += frame.count(label, 0.0)
        return kept, total

    def cell_count(self) -> int:
        return len(self._cells)


class _PreviewDialog(Dialog):
    """Common scaffolding: a recording picker, a run button (off-thread), a status
    line, and a thumbnail grid. Subclasses add config rows + the run callable."""

    def __init__(self, recordings: list, parent=None):
        super().__init__(parent)
        self._recordings = recordings
        self.resize(780, 640)
        layout = self.body()
        form = QFormLayout()
        self._recording = QComboBox()
        for i, recording in enumerate(recordings):
            self._recording.addItem(recording.name, i)
        form.addRow("Recording:", self._recording)
        self._build_form(form)
        layout.addLayout(form)
        self._run_btn = QPushButton(self._run_label())
        self._run_btn.clicked.connect(self._start)
        layout.addWidget(self._run_btn)
        self._status = QLabel("")
        layout.addWidget(self._status)
        # A widget (not a bare layout) with the stretch factor absorbs the slack,
        # so the form stays tight at the top even before the first run.
        self._results_box = QWidget()
        self._results_layout = QVBoxLayout(self._results_box)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._results_box, 1)
        self._results_widget = None
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)

    def _current_recording(self):
        return self._recordings[self._recording.currentData()]

    def _start(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self._run_btn.setEnabled(False)
        self._status.setText("Running…")
        self._worker = _PreviewWorker(
            self._make_fn(self._current_recording()), receiver=self
        )
        self._worker.finished_dir.connect(self._on_done)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_done(self, payload) -> None:
        self._run_btn.setEnabled(True)
        count = self._show_results(payload)
        self._status.setText(self._status_text(count, payload))

    def _status_text(self, count: int, payload) -> str:
        """What the status line says once results are in. Subclasses that return
        something other than an output directory override this."""
        return f"{count} image(s) → {payload}"

    def _set_results(self, widget) -> None:
        if self._results_widget is not None:
            self._results_widget.deleteLater()
        self._results_layout.addWidget(widget)
        self._results_widget = widget

    def _show_results(self, payload) -> int:
        """How this dialog lays its results out, and how many it found.

        No default. There *was* one — a flat grid of JPG thumbnails whose double-click handed
        the file to the OS image viewer — and the only subclass overrides this, so it had never
        run. A base-class default that every subclass replaces is not a default, it is dead code
        wearing one: it went on being read as "the simple case" while nothing exercised it, and
        it was the app's only enlarge affordance, which made the GUI style guide's "no zoom/pan
        in v1" look like a live trade-off rather than a rule about nothing.
        """
        raise NotImplementedError

    def _on_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("")
        QMessageBox.critical(self, self.windowTitle(), message)

    # -- subclass hooks ------------------------------------------------------ #
    def _build_form(self, form: QFormLayout) -> None: ...

    def _run_label(self) -> str:
        return "Run"

    def _validate(self) -> str | None:
        return None

    def _make_fn(self, recording):
        raise NotImplementedError


class PromptTestDialog(_PreviewDialog):
    def __init__(
        self,
        recordings,
        model_id,
        prompt_text,
        compute,
        parent=None,
        model_variant=None,
        closed_set=False,
        classes=None,
    ):
        self._model_id = model_id
        self._model_variant = model_variant
        self._compute = compute
        self._seed_text = prompt_text
        self._closed_set = closed_set
        self._classes = list(classes or [])
        super().__init__(recordings, parent)
        self.setWindowTitle("Detection test" if closed_set else "Prompt test")

    def _build_form(self, form: QFormLayout) -> None:
        if self._closed_set:
            # A closed-set model has no text prompt; pick which of its trained
            # classes to preview (one result column each). Default: all.
            self._class_list = QListWidget()
            for name in self._classes:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self._class_list.addItem(item)
            self._class_list.setFixedHeight(80)
            form.addRow("Classes to show:", self._class_list)
        else:
            self._prompts = QPlainTextEdit(self._seed_text)
            self._prompts.setFixedHeight(80)
            form.addRow("Candidate prompts (one per line):", self._prompts)
        self._n = IntSpin()  # how many frames to sample: this dialog's own knob
        self._n.setRange(1, 100)
        self._n.setValue(6)
        form.addRow("Sample frames:", self._n)
        # No confidence row: the run keeps every proposal and the threshold is chosen on
        # the results below, where you can see what moving it does.

    def _run_label(self) -> str:
        return "Run prompt test"

    def _show_results(self, payload) -> int:
        frames, prompts, colors = payload
        self._confidence = ConfidenceRow()
        self._confidence.threshold_changed.connect(self._on_min_conf)
        self._matrix = _PromptMatrix(frames, prompts, colors, self._confidence.value())
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._confidence)
        column.addWidget(self._matrix, 1)
        self._set_results(holder)
        self._update_caption(self._confidence.value())
        return self._matrix.cell_count()

    def _status_text(self, count: int, payload) -> str:
        frames, prompts, _colors = payload
        return f"{len(frames)} frame(s) × {len(prompts)} prompt(s)"

    def _on_min_conf(self, value: float) -> None:
        self._matrix.set_min_conf(value)
        self._update_caption(value)

    def _update_caption(self, min_conf: float) -> None:
        kept, total = self._matrix.counts(min_conf)
        self._confidence.set_caption(showing_caption(kept, total))

    def _prompt_list(self) -> list[Prompt]:
        lines = [line.strip() for line in self._prompts.toPlainText().splitlines()]
        return [
            Prompt(targets=[PromptTarget(text, TextCue(text))])
            for text in lines
            if text
        ]

    def _checked_classes(self) -> list[str]:
        return [
            self._class_list.item(i).text()
            for i in range(self._class_list.count())
            if self._class_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _validate(self) -> str | None:
        if self._closed_set:
            if not self._checked_classes():
                return "Select at least one class to preview."
            return None
        if not self._prompt_list():
            return "Enter at least one prompt (one per line)."
        return None

    def _make_fn(self, recording):
        n = self._n.value()
        model_id, compute = self._model_id, self._compute
        model_variant = self._model_variant
        # Closed-set: no text prompts; preview the chosen trained classes instead.
        prompts = [] if self._closed_set else self._prompt_list()
        display_classes = self._checked_classes() if self._closed_set else None
        texts = (
            list(display_classes or [])
            if self._closed_set
            else [
                line.strip()
                for line in self._prompts.toPlainText().splitlines()
                if line.strip()
            ]
        )

        def fn():
            # Consume results as they arrive and keep only the compact form: at a wide-open
            # run each frame carries every proposal, and their full-resolution masks would
            # be gigabytes held across the grid.
            frames: list[_CompactFrame] = []
            recording.run_prompt_test(
                model_id=model_id,
                model_variant=model_variant,
                prompts=prompts,
                display_classes=display_classes,
                n_sample_frames=n,
                compute=compute,
                on_result=lambda result: frames.append(_CompactFrame(result)),
            )
            labels = list(frames[0].rows) if frames else []
            prompt_columns = [
                (label, texts[i] if i < len(texts) else label)
                for i, label in enumerate(labels)
            ]
            colors = assign_colors(labels)
            return frames, prompt_columns, colors

        return fn
