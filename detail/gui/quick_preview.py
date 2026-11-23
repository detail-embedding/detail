"""The small player a strip puts in a thumbnail's place: click to play, click to stop.

**Both strips, one widget.** This began inside `cluster_strip` and reached the clustering page
alone, so Query's strip drew stills and Clustering's played — the same split that has produced a
capability-on-one-page four times over (`tests/gui/test_workspace_parity.py`). It lives here so a
third strip inherits it rather than reimplementing it.

**One per strip, not one per cell.** A strip shows up to hundreds of members; a decoder and a
timer each is not affordable, and nobody watches two at once. The strip owns one of these and
moves it into whichever cell was clicked.

**It indexes decoded frames rather than driving a decoder.** That is what keeps it distinct from
`FrameScrubber`, and why it is a permanent entry in that widget's `_PLAYERS` allowlist: the frames
are already in memory, strided to a thumbnail's budget, so there is no rate for a speed menu to be
a multiple of. Double-clicking is the way out of that compromise — see `double_clicked`, which the
host wires to a popup that re-decodes the clip at its real rate.
"""

from detail.gui.busy import BusyOverlay
from detail.gui.qt import (
    QHBoxLayout,
    QLabel,
    QPixmap,
    QPushButton,
    QSlider,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.run_preview import _to_pixmap

#: The thumbnail sizes a strip may ask for. Shared so two strips cannot drift apart on them.
THUMB_PX = 132
MIN_THUMB_PX = 72
MAX_THUMB_PX = 480


class QuickPreview(QWidget):
    """One member, playing: the same crop, animated, over a play/pause and a scrub bar.

    Deliberately those two controls and nothing else. The cell is a member of a set being compared
    against its neighbours, and every control added to it is width taken from the comparison.
    """

    #: Double-clicked, with nothing attached — the strip knows which clip is in here, this
    #: widget only knows the frames it was handed.
    double_clicked = Signal()

    def __init__(self, size_px: int, parent=None) -> None:
        super().__init__(parent)
        self._size_px = int(size_px)
        self._frames: list = []
        self._index = 0
        # The thumbnail's picture, held while the clip decodes so the cell is never blank.
        self._still = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self._image = QLabel("…")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setFixedSize(self._size_px, self._size_px)
        layout.addWidget(self._image)
        # The cell occupies exactly the slot the thumbnail did. Without this the scrub bar's
        # Expanding policy propagates up and the cell eats the free width beside it, so a
        # cluster with two members plays one under a scrub bar four times the picture's width.
        self.setFixedWidth(self._size_px)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(2)
        self._play = QPushButton("▶")
        self._play.setCheckable(True)
        self._play.setFixedWidth(24)
        self._play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play.toggled.connect(self._on_toggled)
        controls.addWidget(self._play)
        self._scrub = QSlider(Qt.Orientation.Horizontal)
        self._scrub.setRange(0, 0)
        self._scrub.valueChanged.connect(self._on_scrubbed)
        controls.addWidget(self._scrub, 1)
        layout.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        # A decode is a second or two of walking video, which the long-op rule says must show
        # itself rather than look like a hang. The scrim is light for the same reason
        # `decode_overlay` keeps it light: the still underneath stays readable while it works.
        self._busy = BusyOverlay(self._image, scrim_alpha=130)

    def set_size(self, size_px: int) -> None:
        self._size_px = int(size_px)
        self._image.setFixedSize(self._size_px, self._size_px)
        self.setFixedWidth(self._size_px)
        if self._frames:
            self._show_frame(self._index)
        elif self._still is not None:
            self._paint(self._still)

    def set_loading(self, rgb=None) -> None:
        """Waiting on the decode — which is not the same as having failed one.

        Clearing the cell by handing it an empty frame list made it announce "could not decode
        this clip" the instant it was clicked, every time, for however long the decode took. A
        placeholder must never claim a failure that has not happened.

        ``rgb`` is the picture the thumbnail was already showing. Decoding a clip is about a
        second of video the machine has to walk through frame by frame, and there is no reason
        to stare at an empty square while it does: the still is the clip's own first frame, so
        the cell looks like the thumbnail it replaced until it starts moving.
        """
        self._frames = []
        self._index = 0
        self._timer.stop()
        self._play.setChecked(False)
        self._play.setEnabled(False)
        self._scrub.setEnabled(False)
        self._scrub.blockSignals(True)
        self._scrub.setRange(0, 0)
        self._scrub.blockSignals(False)
        self._busy.start("Decoding clip…")
        if rgb is None:
            self._image.setPixmap(QPixmap())
            self._image.setText("")
            return
        self._still = rgb
        self._paint(rgb)

    def _set_frames(self, frames, interval_ms: int) -> None:
        """Hand over a **decoded** clip and start it looping.

        Empty here means the decode came back with nothing, which is a real failure — unlike the
        empty state ``set_loading`` puts the cell in while it waits.
        """
        self._busy.stop()
        self._frames = list(frames or [])
        self._index = 0
        self._still = None
        self._play.setEnabled(bool(self._frames))
        self._scrub.setEnabled(bool(self._frames))
        self._timer.setInterval(max(int(interval_ms), 16))
        self._scrub.blockSignals(True)
        self._scrub.setRange(0, max(len(self._frames) - 1, 0))
        self._scrub.setValue(0)
        self._scrub.blockSignals(False)
        if not self._frames:
            self._image.setText("could not decode this clip")
            return
        self._show_frame(0)
        self._play.setChecked(True)

    def _refresh_open_hint(self) -> None:
        """Say the double-click is there, but only where a host has wired it.

        `self.receivers(...)` rather than a flag a caller sets: the claim and the connection
        would otherwise live in two files, and one of them would be wrong.
        """
        self.setToolTip(
            "Double-click to open this clip"
            if self.receivers(self.double_clicked) > 0
            else ""
        )

    def set_frames(self, frames, interval_ms: int) -> None:
        self._refresh_open_hint()
        return self._set_frames(frames, interval_ms)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Out of the thumbnail and into a real player.

        A strided list of frames in a 132px square is a compromise for comparing members side by
        side; the moment somebody wants to *look* at one, it is the wrong thing to be looking at.
        Emitted rather than acted on, because which clip this is showing is the strip's knowledge.
        """
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)

    def stop(self) -> None:
        self._busy.stop()
        self._play.setChecked(False)

    def _on_toggled(self, on: bool) -> None:
        self._play.setText("‖" if on else "▶")
        if on and self._frames:
            # Play from the top when the last press ended at the end, so the button always
            # does something rather than sitting on the final frame.
            if self._index >= len(self._frames) - 1:
                self._show_frame(0)
                self._scrub.blockSignals(True)
                self._scrub.setValue(0)
                self._scrub.blockSignals(False)
            self._timer.start()
        else:
            self._timer.stop()

    def _on_scrubbed(self, value: int) -> None:
        """Dragging the bar pauses, so a frame the user chose is not immediately replaced."""
        if self._timer.isActive():
            self._play.setChecked(False)
        self._show_frame(int(value))

    def _advance(self) -> None:
        if not self._frames:
            return
        # Stops at the end rather than looping. A clip that restarts on its own keeps pulling
        # the eye back to one member while you are trying to look across the row.
        if self._index >= len(self._frames) - 1:
            self._play.setChecked(False)
            return
        self._index += 1
        self._scrub.blockSignals(True)
        self._scrub.setValue(self._index)
        self._scrub.blockSignals(False)
        self._show_frame(self._index)

    def _show_frame(self, index: int) -> None:
        if not self._frames:
            return
        self._index = max(0, min(int(index), len(self._frames) - 1))
        self._paint(self._frames[self._index])

    def _paint(self, rgb) -> None:
        import numpy as np

        self._image.setText("")
        self._image.setPixmap(
            _to_pixmap(np.ascontiguousarray(rgb)).scaled(
                self._size_px,
                self._size_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
