"""Why a scrub position produced no pixels, and the words a preview shows for it.

Every frame-preview surface decodes through a producer that can fail in two ways, and
``storage/video.py::read_frame_at`` keeps them distinct on purpose — its
``FileNotFoundError`` exists, in its own words, "so previews can show a clear message":

- **missing** — the video file is not on this machine (a video-excluded export). The
  fix is to bring the video here; no frame in this recording will ever decode.
- **undecodable** — the file is present but this position yields no picture. Every
  other position may be fine.

A subject crop has a third: **not detected** — the frame decoded, but tracking has no box for
the subject there, so there is nothing to crop. Calling that "could not decode" sent people
checking a video that was fine.

A surface that collapses both into an empty pane reports neither, and one that also
captures input — marking a clip — accepts that input against a frame nobody saw. Route
every scrubber decoder through :func:`decode` and render the reason with
:func:`unavailable_text`.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from detail.api.run_preview import SubjectNotDetectedError

MISSING = "missing"
UNDECODABLE = "undecodable"
NOT_DETECTED = "not_detected"

_TEXT = {
    MISSING: "(source video not in this export)",
    UNDECODABLE: "(could not decode frame)",
    NOT_DETECTED: "(subject not detected in this frame)",
}

T = TypeVar("T")


def unavailable_text(reason: str) -> str:
    """The pane text for a :func:`decode` reason."""
    return _TEXT[reason]


def decode(producer: Callable[[], T | None]) -> tuple[T | None, str | None]:
    """Run a frame producer, classifying the two ways it yields no pixels.

    Runs inside a scrubber decoder on a worker thread, so it stays Qt-free.

    Args:
        producer: Called with no arguments; returns the decoded value, or ``None``
            when this position has no picture.

    Returns:
        ``(value, None)`` on success, ``(None, reason)`` otherwise — ``reason`` is
        :data:`MISSING`, :data:`UNDECODABLE` or :data:`NOT_DETECTED`.
    """
    try:
        value = producer()
    except FileNotFoundError:
        return None, MISSING
    except SubjectNotDetectedError:
        # Before IndexError: both are LookupErrors, and this one is not a decode failure.
        return None, NOT_DETECTED
    except IndexError:
        # ``Recording.get_frame`` reports an undecodable position this way; the
        # preview accessors return None for the same thing.
        return None, UNDECODABLE
    return (value, None) if value is not None else (None, UNDECODABLE)
