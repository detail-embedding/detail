"""Clip thumbnails decoded off the UI thread, one cell at a time, and the score caption rule.

What is left of the clip grid. Its two widgets — the query CALIBRATE's one-row confidence strip
and the grid before it — went when CALIBRATE started playing every clip in the shared montage
(``live_montage``, 2026-11). Two pieces outlived them, because other surfaces use them:

- **``_CellTask``**, the pool decode for one thumbnail, which ``cluster_strip`` decodes through.
  The decode economics it was built for still hold there: progressive rather than batched (a cell
  paints when it lands), latest-wins per cell (``_generation`` drops a result for a band the user
  has left), and a ``QThreadPool`` rather than a thread per cell.
- **``score_digits``**, how many decimals a confidence caption needs to tell its neighbours
  apart, which BROWSE's confidence column and CALIBRATE's captions read.

**A cell that cannot decode says which reason.** ``gui/frame_source.py`` keeps *missing* (the
video is not on this machine) and *undecodable* (this position has no picture) apart, and a
surface that collapses them into a blank pane reports neither; ``_CellTask`` turns any other
failure into *undecodable* rather than leaving a placeholder forever.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from detail.gui import frame_source


class _CellSignals(QObject):
    # (generation, index, rgb | None, reason | None) — delivered on the UI thread.
    done = pyqtSignal(int, int, object, object)


class _CellTask(QRunnable):
    """Decode one cell's thumbnail on a pool worker and hand it back.

    ``signals`` is unparented and shared, the way ``FrameScrubber``'s is: a task holds a
    reference to it, so it outlives a grid whose host closed mid-decode and can still emit
    safely.
    """

    def __init__(
        self, generation: int, index: int, producer, signals: _CellSignals
    ) -> None:
        super().__init__()
        self._generation = generation
        self._index = index
        self._producer = producer
        self._signals = signals

    def run(self) -> None:
        try:
            rgb, reason = frame_source.decode(self._producer)
        except Exception:  # noqa: BLE001 — boundary: a pool worker never raises into Qt
            # `frame_source.decode` classifies the two *expected* failures. Anything else —
            # a recording record that no longer resolves, a corrupt store — would otherwise
            # escape into the thread pool, where it prints a traceback nobody reads and
            # leaves the cell showing its placeholder forever. From the user's side a frame
            # that could not be produced is undecodable, whatever the reason, so it says so.
            rgb, reason = None, frame_source.UNDECODABLE
        try:
            self._signals.done.emit(self._generation, self._index, rgb, reason)
        except RuntimeError:
            pass  # the grid was torn down mid-decode — drop it


#: Scores closer than this, relatively, are one score. The search computes in float32, and
#: how its sums round depends on how BLAS splits them across threads, so one match found in two
#: recordings can come back a unit in the last place apart (0.85658836 and 0.85658842 on a
#: 32-thread machine, equal on a 4-thread one). Counting those as two made every caption take
#: the full five decimals trying to separate them. Ten times float32's resolution, and far
#: below the 1e-5 a five-decimal caption can show.
SAME_SCORE_REL_TOL = 1e-6


def score_digits(scores) -> int:
    """Decimals enough to tell these scores apart, capped so a caption stays a caption.

    Two is what a *cutoff* reads like and it is useless beside a boundary: the matches either
    side of one agree to two places, so a kept clip and a cut one both printed `0.66` and a
    list of them looked like it was not updating.
    """
    values = sorted(v for v in scores if v is not None)
    distinct = sum(
        1
        for i, v in enumerate(values)
        if i == 0 or not math.isclose(v, values[i - 1], rel_tol=SAME_SCORE_REL_TOL)
    )
    digits = 2
    while digits < 5 and len({f"{v:.{digits}f}" for v in values}) < distinct:
        digits += 1
    return digits
