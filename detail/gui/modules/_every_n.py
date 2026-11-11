"""The "Use every N frames" row, shared by the detection / tracking / embedding forms.

One row, one line of feedback: the number of frames to step, and beside it the rate that
lands at — **always**, not only when the run is thinned. A cadence the user cannot see is
one they cannot reason about, and "every 4th frame" means nothing without the source rate
next to it.

Downstream of a thinned run the number is also *constrained*: every stage counts source
frames (§4), so tracking and embedding may only go coarser by a whole multiple of what
they receive. The spin's floor and step are set to the upstream value, so clicking can
only ever land on a reachable number; a typed one is caught by ``warning()``.

The source rate is the container's advertised rate off the (mtime-cached)
``recording.yaml`` — cheap enough to consult per keystroke, and a display number only:
the step itself is an exact integer count either way (Tenet 8).
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from detail.gui._pydantic_form import field_meta
from detail.gui.spin import IntSpin

_MAX = 10_000


def format_rate(fps: float) -> str:
    """A frame rate for a label: whole when it as good as is, else one decimal."""
    return f"{round(fps):g}" if abs(fps - round(fps)) < 0.05 else f"{fps:.1f}"


def source_rate(recordings) -> tuple[float, float] | None:
    """``(min_fps, max_fps)`` across the selection, or ``None`` if no video is known.

    A range rather than a single number because a selection can span rigs recorded at
    different rates, and one of them silently standing for all of them is the kind of
    quiet wrongness the label exists to prevent.
    """
    rates = sorted({r.video_fps for r in recordings if getattr(r, "video_fps", None)})
    return (rates[0], rates[-1]) if rates else None


def describe(every_n: int, upstream: int, rates: tuple[float, float] | None) -> str:
    """The line beside the spin: the resulting rate, then what it keeps of its input.

    ``upstream`` is the stride this stage receives (1 at detection, which reads the video
    itself). The second clause only appears downstream, where "every 4th frame" is
    ambiguous without it — it is either *all* of what the stage above produced, or a
    fraction of it.
    """
    if rates is None:
        return f"every {ordinal(every_n)} frame" if every_n > 1 else "every frame"
    low, high = rates
    if low == high:
        rate = f"{format_rate(low / every_n)} fps (source {format_rate(low)} fps)"
    else:
        rate = (
            f"{format_rate(low / every_n)}–{format_rate(high / every_n)} fps "
            "(sources vary)"
        )
    return rate + _share(every_n, upstream)


def summary(every_n: int, rates: tuple[float, float] | None) -> str:
    """A finished run's cadence, for a read-only view: ``every 4th frame · 7.5 fps
    (source 30 fps)``. Both halves, because neither means much alone."""
    cadence = "every frame" if every_n <= 1 else f"every {ordinal(every_n)} frame"
    if rates is None:
        return cadence
    if every_n <= 1:
        return f"{cadence} · {format_rate(rates[0])} fps"
    return f"{cadence} · {describe(every_n, 1, rates)}"


def _share(every_n: int, upstream: int) -> str:
    """`` · every detected frame`` / `` · half the detected frames``, or nothing.

    Only meaningful downstream (``upstream > 1``); at detection the input is the video and
    the rate alone says everything.
    """
    if upstream <= 1:
        return ""
    factor = every_n // upstream
    noun = "detected" if upstream else ""
    if factor <= 1:
        return f" · every {noun} frame"
    phrases = {
        2: f"half the {noun} frames",
        3: f"a third of the {noun} frames",
        4: f"a quarter of the {noun} frames",
    }
    return " · " + phrases.get(factor, f"one in {factor} {noun} frames")


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


class EveryNRow(QWidget):
    """``[ 4 ⌃⌄] frames    7.5 fps (source 30 fps) · every detected frame``.

    ``upstream_noun`` names the stage above ("detected" / "tracked") for the share clause.
    ``changed`` fires whenever the cadence moves, for a form with another readout that
    depends on it (tracking's frames-counted tolerance).
    """

    changed = pyqtSignal()

    def __init__(self, model, *, upstream_noun: str = "", parent=None) -> None:
        super().__init__(parent)
        meta = field_meta(model, "every_n_frames")
        self._noun = upstream_noun
        self._upstream = 1
        self._rates: tuple[float, float] | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._spin = IntSpin()
        self._spin.setRange(1, _MAX)
        self._spin.setSuffix(" frames")
        self._spin.setToolTip(meta["help"])
        self._spin.valueChanged.connect(self._sync)
        layout.addWidget(self._spin)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        layout.addWidget(self._note, 1)
        self._sync()

    # -- state ------------------------------------------------------------- #
    def set_recordings(self, recordings) -> None:
        """Refresh the source rate from the selection — reads ``video_fps`` per recording."""
        self.set_rates(source_rate(recordings))

    def set_rates(self, rates: tuple[float, float] | None) -> None:
        """Install an already-measured rate range.

        For a caller that has resolved the selection off the UI thread: ``source_rate`` costs
        a ``recording.yaml`` round trip per recording, and a form that also asks for it in its
        own notes pays for it once per asker. The tracking form paid three times a click.
        """
        self._rates = rates
        self._sync()

    def set_upstream(self, upstream: int | None) -> None:
        """Adopt the cadence of the run this stage reads.

        The floor and the step both become ``upstream``: a finer value asks for frames
        that were never processed, and a non-multiple asks for frames that fall between
        the ones that were. Pre-fills the spin when it is still sitting on a value the
        new upstream cannot produce.
        """
        upstream = max(1, int(upstream or 1))
        self._upstream = upstream
        self._spin.setMinimum(upstream)
        self._spin.setSingleStep(upstream)
        if self._spin.value() % upstream:
            self._spin.setValue(upstream)
        self._sync()

    def value(self) -> int:
        return self._spin.value()

    def set_value(self, every_n: int | None) -> None:
        self._spin.setValue(max(self._upstream, int(every_n or self._upstream)))

    def warning(self) -> str | None:
        """Why a **typed** value is unreachable, for the form's ``validation_error``."""
        every_n = self._spin.value()
        if self._upstream > 1 and every_n % self._upstream:
            reachable = ", ".join(str(self._upstream * n) for n in (1, 2, 3))
            return (
                f"the run above used every {ordinal(self._upstream)} frame — "
                f"use {reachable}, …"
            )
        return None

    # -- rendering --------------------------------------------------------- #
    def _render(self) -> None:
        problem = self.warning()
        if problem is not None:
            self._note.setText(f"⚠ {problem}")
            return
        text = describe(self._spin.value(), self._upstream, self._rates)
        if self._noun:
            text = text.replace("detected", self._noun)
        self._note.setText(text)

    def _sync(self) -> None:
        self._render()
        self.changed.emit()
