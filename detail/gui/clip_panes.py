"""The clip popup's picture: the footage large on the left, the masks alone and the pose alone
stacked on the right (``ClipQueryPreview.clip_views``, drawn by ``api.clip_views``).

**Why three pictures rather than three overlays.** Telling one behaviour from another in a
cluster is a question about posture and contact, and the footage hides both under fur, shadow
and the other animal. The masks on white show the outlines and whether the animals touch; the
pose on white shows the body's geometry with nothing else in the frame. Drawn *over* the footage
the same marks compete with what they are meant to explain, so the footage keeps its own toggles
for that and the two side panes are never drawn on it.

**Placed, not laid out.** The side panes are half the footage's height each, so the picture is
1.5 frames wide by one tall (:meth:`ClipPanes.aspect_size`, which the player hands its
``PictureWithControls`` so the transport sits directly under it). A stretch grid split the width
by ratio whatever the frame's shape and letterboxed the footage inside its column, leaving a gap
between it and the panes beside it; here the three rectangles are computed from the frame's own
aspect ratio and centred as one block, the way ``PictureWithControls`` places its pair.

**Click a side pane to enlarge it** (``enlarge_wanted``): a pane is a half-height thumbnail of a
full-resolution picture, and studying an outline or a gait wants it large. The cue — a pointing
hand and "Click to enlarge" — is **derived from the wiring**: it appears only while something is
connected to ``enlarge_wanted``, so a host that does not open the window never advertises it.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QWidget

from detail.gui.image_view import ImageView
from detail.gui.run_preview import _to_pixmap

#: Pixels between the footage and the side column, and between the two side panes.
_GAP = 6
#: The two side panes, as ``enlarge_wanted`` names them (``api.clip_views.PANES``).
from detail.api.clip_views import KEYPOINTS, MASKS  # noqa: E402

ENLARGE_TIP = (
    "Click to enlarge: this pane at full size, with its own playback and export."
)


class ClipPanes(QWidget):
    """``frame`` (the player's own view, kept as it is: click-to-play, zoom, the decode scrim)
    beside the masks pane over the pose pane, each side pane under a one-word caption.
    """

    #: A side pane was clicked: ``"masks"`` or ``"keypoints"``.
    enlarge_wanted = pyqtSignal(str)

    def __init__(self, frame: ImageView, parent=None) -> None:
        super().__init__(parent)
        self.frame = frame
        self.masks = ImageView(zoomable=False)
        self.pose = ImageView(zoomable=False)
        self.masks.setObjectName("clip_masks_pane")
        self.pose.setObjectName("clip_pose_pane")
        self._captions = (
            QLabel("Masks"),
            QLabel("Keypoints"),
        )
        self._captions[0].setToolTip(
            "Every tracked animal's mask on white: the clip's animal in orange, the others "
            "grey."
        )
        self._captions[1].setToolTip(
            "The keypoints the embedding run used, on white: the clip's animal in colour, "
            "the others grey."
        )
        self._caption_tips = tuple(c.toolTip() for c in self._captions)
        #: What the pose pane says about itself (why there are no keypoints), kept apart from
        #: the enlarge cue so one never overwrites the other.
        self._pose_note = ""
        self.masks.clicked.connect(lambda: self._enlarge(MASKS))
        self.pose.clicked.connect(lambda: self._enlarge(KEYPOINTS))
        for child in (frame, self.masks, self.pose, *self._captions):
            child.setParent(self)
            child.show()
        self._aspect = 16 / 9  # the footage's, until a frame says otherwise
        self.setMinimumHeight(frame.minimumHeight())
        self._refresh_cue()

    # -- enlarging ------------------------------------------------------------ #
    def enlargeable(self) -> bool:
        """Whether a click on a side pane opens anything: something is listening."""
        return self.receivers(self.enlarge_wanted) > 0

    def _enlarge(self, pane: str) -> None:
        if self.enlargeable():
            self.enlarge_wanted.emit(pane)

    def connectNotify(self, signal) -> None:  # noqa: N802 — Qt's name
        super().connectNotify(signal)
        self._refresh_cue()

    def disconnectNotify(self, signal) -> None:  # noqa: N802 — Qt's name
        super().disconnectNotify(signal)
        self._refresh_cue()

    def _refresh_cue(self) -> None:
        """The pointing hand and "Click to enlarge", exactly while a click opens something."""
        if not hasattr(self, "_caption_tips"):
            return  # connectNotify during construction, before the panes exist
        wired = self.enlargeable()
        shape = Qt.CursorShape.PointingHandCursor if wired else None
        for pane, caption, tip, note in (
            (self.masks, self._captions[0], self._caption_tips[0], ""),
            (self.pose, self._captions[1], self._caption_tips[1], self._pose_note),
        ):
            pane.set_rest_cursor(shape)
            pane.setToolTip(
                " ".join(t for t in (note, ENLARGE_TIP if wired else "") if t)
            )
            caption.setToolTip(f"{tip} {ENLARGE_TIP}" if wired else tip)

    def set_frame_aspect(self, size: QSize | None) -> None:
        """Adopt a decoded frame's shape, re-placing the panes if it changed."""
        if size is None or size.height() <= 0:
            return
        aspect = size.width() / size.height()
        if aspect != self._aspect:
            self._aspect = aspect
            self._place()

    def show_views(self, views) -> None:
        """Put one ``ClipViews`` on the two side panes (the frame is the player's to set)."""
        self.masks.set_frame(_to_pixmap(np.ascontiguousarray(views.masks)))
        if views.pose is None:
            # The pane is a picture, so no word wrap: a short line here, the whole reason on
            # hover, as the greyed Keypoints tick also gives it.
            self.pose.clear_frame("No keypoints for this run")
            note = views.pose_note or ""
        else:
            self.pose.set_frame(_to_pixmap(np.ascontiguousarray(views.pose)))
            note = ""
        if note != self._pose_note:
            self._pose_note = note
            self._refresh_cue()

    def clear(self, text: str) -> None:
        """Nothing to show on the side panes (an undecodable frame)."""
        self.masks.clear_frame(text)
        self.pose.clear_frame(text)

    @staticmethod
    def aspect_size(frame_size: QSize | None) -> QSize | None:
        """The whole picture's shape for a frame of ``frame_size``: the side column is half a
        frame wide, since each of its panes is half a frame tall."""
        if frame_size is None:
            return None
        return QSize(int(round(frame_size.width() * 1.5)), frame_size.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().resizeEvent(event)
        self._place()

    def _place(self) -> None:
        width, height = max(self.width(), 1), max(self.height(), 1)
        whole = 1.5 * self._aspect
        if (
            width / height > whole
        ):  # wider than the picture: full height, centred across
            tall = height
            left, top = int((width - whole * tall) / 2), 0
        else:  # narrower: full width, centred down
            tall = int(width / whole)
            left, top = 0, int((height - tall) / 2)
        frame_w = int(self._aspect * tall)
        self.frame.setGeometry(QRect(left, top, frame_w, tall))
        side_x = left + frame_w + _GAP
        side_w = max(int(whole * tall) - frame_w - _GAP, 1)
        cell = (tall - _GAP) // 2
        for row, (caption, pane) in enumerate(
            zip(self._captions, (self.masks, self.pose))
        ):
            y = top + row * (cell + _GAP)
            caption_h = caption.sizeHint().height()
            caption.setGeometry(QRect(side_x, y, side_w, caption_h))
            pane.setGeometry(
                QRect(side_x, y + caption_h, side_w, max(cell - caption_h, 1))
            )
