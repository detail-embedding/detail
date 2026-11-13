"""A frame from a recording, cropped, to draw a skeleton over (§26 *The Keypoints manager*).

The skeleton editor draws on a plain canvas because a skeleton belongs to no recording. Placing
points on a body you can see is easier than on nothing, so :class:`FramePickerDialog` lends the
canvas a picture: pick one of the open experiment's recordings, play or scrub it with the app's
own player (``FrameScrubber``, decoding off the UI thread like every other frame view), drag a
rectangle on the frame to crop it, and **Use this frame**.

The picture is a drawing aid and nothing more. It is never saved: the skeleton's ``layout`` is
fractions of its canvas, and :func:`square_backdrop` pads the crop to the canvas's square so those
fractions mean the same thing with the picture as without it. A skeleton drawn over a long, thin
crop reopens on the plain canvas in the same proportions, not stretched.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from detail.gui import frame_source
from detail.gui.arena import ArenaCanvas
from detail.gui.busy import decode_overlay
from detail.gui.dialog import Dialog
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines, set_available
from detail.gui.workers import BackgroundCall

#: What the canvas holds until the first frame decodes, behind the "Decoding frame…" overlay.
_PLACEHOLDER = np.zeros((360, 480, 3), np.uint8)
#: The plain canvas's grey (``skeleton_editor.BLANK``), so the padding around a crop reads as
#: the canvas rather than as part of the picture.
PAD_VALUE = 56
HINT = (
    "drag on the frame: crop · drag a corner: resize · drag inside: move · scroll: zoom · "
    "right-drag: pan"
)
_NO_FRAME = "No frame on screen yet."


def square_backdrop(image: np.ndarray) -> np.ndarray:
    """``image`` centred on a square of the plain canvas's grey, its longer side long."""
    height, width, _ = image.shape
    side = max(height, width)
    out = np.full((side, side, 3), PAD_VALUE, dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    out[top : top + height, left : left + width] = image
    return out


def crop_box(
    crop: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """A normalized ``(x0, y0, x1, y1)`` crop as whole pixels of a ``width`` × ``height``
    frame: outward to the pixels it touches, and never empty."""
    x0, y0, x1, y1 = crop
    left = min(max(math.floor(x0 * width), 0), width - 1)
    top = min(max(math.floor(y0 * height), 0), height - 1)
    right = min(max(math.ceil(x1 * width), left + 1), width)
    bottom = min(max(math.ceil(y1 * height), top + 1), height)
    return left, top, right, bottom


def _locate(part_counts: list[int], frame: int) -> tuple[int, int]:
    for part, count in enumerate(part_counts):
        if frame < count:
            return part, frame
        frame -= count
    return len(part_counts) - 1, part_counts[-1] - 1


def _video_of(recording) -> tuple[list[int], np.ndarray, float | None]:
    """What the player needs to open ``recording``: the frames in each part, the timestamp
    map and the rate. Each reads the timestamp maps, so this runs off the UI thread."""
    times = recording.frame_timestamps()  # off-thread: run by a BackgroundCall
    return recording.part_frame_counts, times, recording.video_fps


def _decode_frame(recording, part_counts: list[int], frame: int):
    """The scrubber's decoder for one recording: ``(recording id, image, reason)``.

    Bound to its recording, so a frame of the last recording landing after a switch says
    which recording it is from and is dropped rather than shown under the new one's name.
    """
    part, local = _locate(part_counts, frame)
    image, reason = frame_source.decode(lambda: recording.get_frame(part, local))
    return recording.id, image, reason


class CropCanvas(ArenaCanvas):
    """A frame with a crop rectangle: drag on the frame to draw one, drag a corner to resize
    it, drag inside it to move it.

    The scene-geometry canvas underneath, so zoom (scroll) and pan (right-drag) are the same
    gestures as on every other frame you draw on. The rectangle is normalized ``(x0, y0, x1,
    y1)``, so it stays where it was on a frame of another size.
    """

    #: The rectangle was drawn, moved, resized or cleared.
    crop_changed = pyqtSignal()
    #: A drag shorter than this (widget pixels) is a click, which leaves the crop as it was.
    _MIN_DRAG_PX = 4
    _SHADE = QColor(0, 0, 0, 140)
    _EDGE = QColor(255, 255, 255)
    _HALO = QColor(20, 20, 20, 200)

    def __init__(self) -> None:
        super().__init__(_PLACEHOLDER, points=[], max_points=None)
        self._crop: tuple[float, float, float, float] | None = None
        # The drag in flight: ("resize", fixed corner) or ("move", offset of the press from
        # the crop's top-left), both normalized; ("new", press point) until it passes the slop.
        self._grab: tuple[str, tuple[float, float]] | None = None
        self._press: QPointF | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def crop(self) -> tuple[float, float, float, float] | None:
        return self._crop

    def set_crop(self, crop: tuple[float, float, float, float] | None) -> None:
        if crop is not None:
            x0, y0, x1, y1 = crop
            crop = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if crop[0] == crop[2] or crop[1] == crop[3]:
                crop = None
        changed = crop != self._crop
        self._crop = crop
        self.update()
        if changed:
            self.crop_changed.emit()

    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` of the frame on the canvas."""
        return self._width, self._height

    def _crop_rect(self, rect: QRectF) -> QRectF:
        x0, y0, x1, y1 = self._crop
        return QRectF(
            QPointF(rect.x() + x0 * rect.width(), rect.y() + y0 * rect.height()),
            QPointF(rect.x() + x1 * rect.width(), rect.y() + y1 * rect.height()),
        )

    def _corner_at(self, pos: QPointF) -> tuple[float, float] | None:
        """The corner *opposite* the one under ``pos`` (what a resize holds still), or None."""
        if self._crop is None:
            return None
        x0, y0, x1, y1 = self._crop
        rect = self._image_rect()
        for (cx, cy), opposite in (
            ((x0, y0), (x1, y1)),
            ((x1, y0), (x0, y1)),
            ((x0, y1), (x1, y0)),
            ((x1, y1), (x0, y0)),
        ):
            wx = rect.x() + cx * rect.width()
            wy = rect.y() + cy * rect.height()
            if (pos.x() - wx) ** 2 + (pos.y() - wy) ** 2 <= self._HIT_RADIUS**2:
                return opposite
        return None

    def mousePressEvent(self, event) -> None:
        if self._unavailable is not None:
            return
        if event.button() == Qt.MouseButton.RightButton or self._pan_anchor is not None:
            super().mousePressEvent(
                event
            )  # a right drag pans; there is no point to remove
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        opposite = self._corner_at(pos)
        if opposite is not None:
            self._grab = ("resize", opposite)
        elif self._crop is not None and self._crop_rect(self._image_rect()).contains(
            pos
        ):
            x, y = self._normalized(pos)
            self._grab = ("move", (x - self._crop[0], y - self._crop[1]))
        elif self._image_rect().contains(pos):
            self._grab = ("new", self._normalized(pos))
            self._press = pos

    def mouseMoveEvent(self, event) -> None:
        if self._grab is None or self._pan_anchor is not None:
            super().mouseMoveEvent(event)
            return
        pos = event.position()
        kind, anchor = self._grab
        x, y = self._normalized(pos)
        if kind == "move":
            x0, y0, x1, y1 = self._crop
            w, h = x1 - x0, y1 - y0
            left = min(max(x - anchor[0], 0.0), 1.0 - w)
            top = min(max(y - anchor[1], 0.0), 1.0 - h)
            self._crop = (left, top, left + w, top + h)
        else:
            if kind == "new":
                if (pos - self._press).manhattanLength() <= self._MIN_DRAG_PX:
                    return
                self._grab = ("resize", anchor)
            self._crop = (
                min(anchor[0], x),
                min(anchor[1], y),
                max(anchor[0], x),
                max(anchor[1], y),
            )
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._grab is None:
            super().mouseReleaseEvent(event)
            return
        kind = self._grab[0]
        self._grab = None
        self._press = None
        if kind == "new":
            return  # a click, not a drag: the crop stays as it was
        crop = self._crop
        if crop is not None and (crop[0] == crop[2] or crop[1] == crop[3]):
            self._crop = None
        self.update()
        self.crop_changed.emit()

    def paintEvent(self, event) -> None:
        if self._unavailable is not None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._image_rect()
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        if self._crop is None:
            return
        crop = self._crop_rect(rect)
        outside = QPainterPath()
        outside.addRect(rect)
        inside = QPainterPath()
        inside.addRect(crop)
        painter.fillPath(outside.subtracted(inside), self._SHADE)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.setPen(QPen(self._HALO, 3))
        painter.drawRect(crop)
        painter.setPen(QPen(self._EDGE, 1.5))
        painter.drawRect(crop)
        painter.setBrush(QBrush(self._EDGE))
        painter.setPen(QPen(self._HALO, 1))
        for corner in (
            crop.topLeft(),
            crop.topRight(),
            crop.bottomLeft(),
            crop.bottomRight(),
        ):
            painter.drawRect(QRectF(corner.x() - 4, corner.y() - 4, 8, 8))


class FramePickerDialog(Dialog):
    """Pick a recording of ``experiment``, play it to a frame, crop it, and take it.

    :meth:`image` is the cropped RGB frame once **Use this frame** closed the dialog (None
    otherwise), and :meth:`source` says where it came from. The recordings are listed by name,
    read off the UI thread; opening one reads its timestamp maps off the UI thread too.
    """

    def __init__(self, experiment, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Background from a recording")
        self._experiment = experiment
        self._recording = None
        self._shown: np.ndarray | None = None
        self._shown_frame: int | None = None
        self._image: np.ndarray | None = None
        self._source = ""

        body = self.body()
        intro = QLabel(
            "Play or scrub to the frame you want, drag on it to crop, then Use this frame. "
            "It goes behind the skeleton canvas while you draw, and is not saved with the "
            "skeleton."
        )
        intro.setWordWrap(True)
        body.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Recording:"))
        self._recordings = QComboBox()
        self._recordings.setMinimumWidth(lines(16))
        self._recordings.activated.connect(self._choose)
        row.addWidget(self._recordings, 1)
        body.addLayout(row)

        self._canvas = CropCanvas()
        self._canvas.crop_changed.connect(self._sync)
        body.addWidget(self._canvas, 1)
        hint = QLabel(HINT)
        hint.setWordWrap(True)
        body.addWidget(hint)
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.rendered.connect(self._on_rendered)
        self._decode_busy = decode_overlay(self._scrubber, self._canvas)
        body.addWidget(self._scrubber)

        crop_row = QHBoxLayout()
        self._crop_label = QLabel("")
        crop_row.addWidget(self._crop_label)
        self._whole = QPushButton("Whole frame")
        self._whole.setAutoDefault(False)
        self._whole.clicked.connect(lambda: self._canvas.set_crop(None))
        crop_row.addWidget(self._whole)
        crop_row.addStretch()
        body.addLayout(crop_row)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._use = box.button(QDialogButtonBox.StandardButton.Ok)
        self._use.setText("Use this frame")
        box.accepted.connect(self.use_frame)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

        self._unavailable("Reading the experiment's recordings…")
        self._names_call = BackgroundCall(experiment.recording_names, receiver=self)
        self._names_call.done.connect(self._listed)
        self._names_call.failed.connect(
            lambda exc: self._unavailable(f"(could not read the recordings: {exc})")
        )
        self._names_call.start()

    # -- public -------------------------------------------------------------- #
    def canvas(self) -> CropCanvas:
        return self._canvas

    def scrubber(self) -> FrameScrubber:
        return self._scrubber

    def recording_id(self) -> str | None:
        return None if self._recording is None else self._recording.id

    def image(self) -> np.ndarray | None:
        """The cropped frame taken by **Use this frame**, or None."""
        return self._image

    def source(self) -> str:
        """Where :meth:`image` came from: ``frame 120 of fish_A1, 200 × 80 px``."""
        return self._source

    def select_recording(self, recording_id: str) -> None:
        """Open ``recording_id`` in the player, as picking it from the list does."""
        index = self._recordings.findData(recording_id)
        if index >= 0:
            self._recordings.setCurrentIndex(index)
            self._choose(index)

    def use_frame(self) -> None:
        """Take the frame on screen, cropped, and close."""
        if self._shown is None:
            return
        self._scrubber.pause()
        height, width, _ = self._shown.shape
        left, top, right, bottom = (
            crop_box(self._canvas.crop(), width, height)
            if self._canvas.crop() is not None
            else (0, 0, width, height)
        )
        self._image = np.ascontiguousarray(self._shown[top:bottom, left:right])
        self._source = (
            f"frame {self._shown_frame} of {self._recordings.currentText()}, "
            f"{right - left} × {bottom - top} px"
        )
        self.accept()

    # -- the recording ----------------------------------------------------------- #
    def _unavailable(self, text: str) -> None:
        self._shown = None
        self._canvas.set_unavailable(text)
        self._scrubber.set_available(False, text)
        self._sync()

    def _listed(self, names: dict[str, str]) -> None:
        self._recordings.clear()
        for recording_id, name in names.items():
            self._recordings.addItem(name, recording_id)
        set_available(
            self._recordings,
            bool(names),
            "This experiment has no recordings yet.",
        )
        if not names:
            self._unavailable("This experiment has no recordings yet.")
            return
        self._choose(0)

    def _choose(self, index: int) -> None:
        recording_id = self._recordings.itemData(index)
        if recording_id is None:
            return
        recording = self._experiment.get_recording(recording_id)
        self._recording = recording
        self._scrubber.set_total(0)
        self._unavailable("Opening the video…")
        self._open_call = BackgroundCall(
            functools.partial(_video_of, recording), receiver=self
        )
        self._open_call.done.connect(
            lambda result, rid=recording_id: self._opened(rid, result)
        )
        self._open_call.failed.connect(
            lambda exc, rid=recording_id: self._open_failed(rid, exc)
        )
        self._open_call.start()

    def _open_failed(self, recording_id: str, exc: BaseException) -> None:
        if recording_id == self.recording_id():
            self._unavailable(f"(could not open the video: {exc})")

    def _opened(self, recording_id: str, result) -> None:
        if recording_id != self.recording_id():
            return  # another recording was picked while this one opened
        part_counts, t_ns, fps = result
        total = sum(part_counts)
        if not total:
            self._unavailable("No video imported")
            return
        self._scrubber.set_decoder(
            functools.partial(_decode_frame, self._recording, list(part_counts))
        )
        if fps:
            self._scrubber.set_playback_rate(fps)
        self._scrubber.set_total(total)
        if len(t_ns) == total:
            self._scrubber.set_timestamps(t_ns)
        self._scrubber.set_available(True, "")
        self._scrubber.set_frame(0, decode=True)

    def _on_rendered(self, frame: int, payload) -> None:
        recording_id, image, reason = payload
        if recording_id != self.recording_id():
            return  # a frame of the recording shown before this one
        if reason is not None:
            self._shown = None
            self._canvas.set_unavailable(frame_source.unavailable_text(reason))
        else:
            self._shown = image
            self._shown_frame = frame
            self._canvas.set_image(image)
        self._sync()

    def _sync(self) -> None:
        crop = self._canvas.crop()
        if crop is None:
            self._crop_label.setText("Crop: the whole frame (drag on it to crop)")
        else:
            width, height = self._canvas.frame_size()
            left, top, right, bottom = crop_box(crop, width, height)
            self._crop_label.setText(
                f"Crop: {right - left} × {bottom - top} px at ({left}, {top})"
            )
        set_available(self._whole, crop is not None, "Already the whole frame.")
        set_available(self._use, self._shown is not None, _NO_FRAME)
