"""A drag that reports while it moves: at most every 50 ms, then once more on release.

Two filter gestures drag (a range's handles on its histogram, a box on the location map), and the
page shades the tree on every change. Emitting on release alone left the tree still while the
hand moved, which reads as nothing happening; emitting on every mouse move recomputes the shading
dozens of times a second for positions the hand has already left. One rule for both, here.
"""

from __future__ import annotations

from collections.abc import Callable
import time

#: The shortest gap between two live emits during a drag.
INTERVAL_S = 0.05


class Throttle:
    """Calls ``emit(value)`` for a changed value at most every :data:`INTERVAL_S` while
    :meth:`offer` is fed, and for the last value on :meth:`flush` if it was not sent yet.
    """

    def __init__(self, emit: Callable[[object], None], interval_s: float = INTERVAL_S):
        self._emit = emit
        self._interval = interval_s
        self._at = float("-inf")
        self._sent: object = None
        self._has_sent = False

    def start(self, value: object) -> None:
        """A drag began at ``value``, which is already shown: not sent, and not counted
        against the interval, so the first real move goes out at once."""
        self._at = float("-inf")
        self._sent = value
        self._has_sent = True

    def offer(self, value: object) -> None:
        """A drag moved to ``value``: sent now unless the last send was too recent."""
        if self._has_sent and value == self._sent:
            return
        now = time.monotonic()
        if now - self._at >= self._interval:
            self._send(value, now)

    def flush(self, value: object) -> None:
        """The drag ended at ``value``: sent unless it is what was sent last. Starts afresh."""
        if not (self._has_sent and value == self._sent):
            self._send(value, time.monotonic())
        self._at = float("-inf")
        self._has_sent = False
        self._sent = None

    def _send(self, value: object, now: float) -> None:
        self._at = now
        self._sent = value
        self._has_sent = True
        self._emit(value)
