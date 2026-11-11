"""One horizontal zoom for every widget that has one: the window, its overview, its gestures.

The cluster tree and the ethograms each carried their own copy — a window state, a wheel mapping,
a pinch, a hand-drawn overview band with its own press-and-drag — and the overviews looked like
nothing else in the app. Two copies were tolerable; the third zoomable widget would have been one
more. So the model is here once:

- **`ZoomWindow`** is the visible part of a span, as fractions ``(left, right)`` of it. Clamped to
  the span, never narrower than its floor, and exactly ``(0, 1)`` when it shows everything.
- **`OverviewBar`** is that window as a real horizontal `QScrollBar`: the handle is the window,
  the track is the whole span. Being a scrollbar it takes the app's style (`gui/style.py`), so it
  is the same control as every other scrollbar in the tool, not a picture of one.
- **`zoom_wheel`** / **`zoom_pinch`** are the gesture mapping: the wheel zooms around the pointer,
  a sideways scroll pans, a pinch zooms. Shift+wheel stays with each widget, because what it moves
  through differs — the tree's levels, the ethogram's recordings.

`requirements.py`'s `zoom_lives_in_one_place` holds the package to it: a class that zooms on the
wheel uses a `ZoomWindow`, or is listed there with the reason it cannot.
"""

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QScrollBar, QSizePolicy

#: A wheel notch scales the visible part by this.
WHEEL_STEP = 0.8
#: Movement past this many pixels turns a press into a pan rather than a click.
DRAG_PX = 4
#: A sideways scroll notch pans by this share of the visible part.
PAN_STEP = 0.1


class ZoomWindow(QObject):
    """The visible part of a span, as fractions ``(left, right)``; ``(0, 1)`` is all of it."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._view = (0.0, 1.0)
        self._floor = 0.0

    @property
    def view(self) -> tuple[float, float]:
        return self._view

    @property
    def size(self) -> float:
        return self._view[1] - self._view[0]

    def is_zoomed(self) -> bool:
        return self._view != (0.0, 1.0)

    def set_floor(self, fraction: float) -> None:
        """The narrowest the window may get, as a fraction of the span."""
        self._floor = min(max(float(fraction), 0.0), 1.0)
        if self.is_zoomed() and self.size < self._floor:
            left, _right = self._view
            self.set_view(left, left + self._floor)

    def set_view(self, left: float, right: float) -> None:
        """Show ``(left, right)``, kept inside the span and no narrower than the floor."""
        size = min(max(right - left, self._floor), 1.0)
        left = min(max(left, 0.0), 1.0 - size)
        view = (0.0, 1.0) if size >= 1.0 else (left, left + size)
        if view != self._view:
            self._view = view
            self.changed.emit()

    def fit(self) -> None:
        """All of the span again."""
        self.set_view(0.0, 1.0)

    def zoom_by(self, factor: float, anchor: float = 0.5) -> None:
        """Scale the window by ``factor`` (< 1 zooms in), keeping the point ``anchor`` of the way
        across the *visible* part where it is — what makes a wheel zoom feel like moving toward
        something rather than the picture sliding away."""
        anchor = min(max(anchor, 0.0), 1.0)
        left, right = self._view
        at = left + anchor * (right - left)
        size = min(max((right - left) * factor, self._floor), 1.0)
        self.set_view(at - anchor * size, at - anchor * size + size)

    def pan_by(self, share: float) -> None:
        """Move by ``share`` of the visible part (positive is later)."""
        left, right = self._view
        shift = share * (right - left)
        self.set_view(left + shift, right + shift)

    def centre_on(self, fraction: float) -> None:
        """Put ``fraction`` of the span at the window's centre, keeping its size."""
        size = self.size
        self.set_view(fraction - size / 2, fraction + size / 2)


class OverviewBar(QScrollBar):
    """A `ZoomWindow` as a horizontal scrollbar: the handle is what is on screen, the track is all
    of it.

    **Hidden at the whole span, with its space kept** (``retain_space``). Greyed, an empty bar
    drew as an outlined box with no handle — a control with nothing in it. Hidden without
    keeping its space, it would resize the picture under the pointer on the first zoom, the
    reason the tree's vertical bar is greyed rather than hidden. A host whose row has a fixed
    height of its own can pass ``retain_space=False`` and give the room to something else.
    """

    #: Steps across the span. Fine enough that a 2-second window on a 2-hour axis still moves.
    RESOLUTION = 100_000

    def __init__(
        self, window: ZoomWindow, parent=None, *, retain_space: bool = True
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._window = window
        self._syncing = False
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        policy.setRetainSizeWhenHidden(retain_space)
        self.setSizePolicy(policy)
        self.setToolTip("Where you are in the whole span; drag it to move")
        window.changed.connect(self._sync)
        self.valueChanged.connect(self._moved)
        self._sync()

    def _sync(self) -> None:
        left, right = self._window.view
        steps = self.RESOLUTION
        self._syncing = True
        self.setPageStep(max(int(round((right - left) * steps)), 1))
        self.setRange(0, steps - self.pageStep())
        self.setSingleStep(max(self.pageStep() // 10, 1))
        self.setValue(int(round(left * steps)))
        self._syncing = False
        self.setVisible(self._window.is_zoomed())

    def _moved(self, value: int) -> None:
        if self._syncing:
            return
        left = value / self.RESOLUTION
        self._window.set_view(left, left + self._window.size)


def zoom_wheel(window: ZoomWindow, event, anchor: float) -> bool:
    """The wheel zooms around ``anchor`` (the pointer's share of the visible width); a sideways
    scroll pans. Returns whether the event was one of those. Shift is the caller's."""
    delta = event.angleDelta()
    if delta.y():
        window.zoom_by(WHEEL_STEP ** (delta.y() / 120.0), anchor)
        return True
    if delta.x():
        window.pan_by(-delta.x() / 120.0 * PAN_STEP)
        return True
    return False


def zoom_pinch(window: ZoomWindow, event, anchor: float) -> bool:
    """A trackpad pinch zooms around ``anchor``. Returns whether ``event`` was one."""
    if (
        event.type() == QEvent.Type.NativeGesture
        and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
    ):
        # value() is the scale change since the last event.
        window.zoom_by(1.0 / (1.0 + event.value()), anchor)
        return True
    return False
