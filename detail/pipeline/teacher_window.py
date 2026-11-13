"""Keypoint teacher runs: detection, tracking and keypoints over a recording's first minutes.

A teacher (§26, #717) is built from one short window of each acquisition setup, so DINOv3
runs on minutes rather than hours. Detection takes the window (``teacher_window_s``) and reads
only the frames before it; tracking and keypoints inherit it from their upstream run and record
it, the way they inherit ``every_n_frames``.

A windowed run covers a prefix of its recording, so it is a teacher run and nothing else:
:data:`TEACHER_CONSUMERS` may take one as input, and every other consumer refuses it through
:func:`refuse_teacher_run`. That refusal is the point — an embedding, a location table or an
export over five minutes of a two-hour recording would be well formed and wrong.
"""

WINDOW_KEY = "teacher_window_s"

#: The processing modules that may consume a teacher run: the chain that builds the teacher,
#: which for several animals merges each one's track (``track_merge``). Each inherits the window
#: and records it, so its own run is a teacher run in turn. Keypoint references, audits and the
#: keypoint distill read teacher runs too, through the API rather than as a module's upstream.
TEACHER_CONSUMERS = frozenset(
    {"tracking", "track_merge", "keypoint_detect", "keypoint_track"}
)


def window_of(record: dict | None) -> float | None:
    """The teacher window a run record carries, in seconds, or None for a whole-recording run."""
    if not record:
        return None
    window = record.get(WINDOW_KEY)
    return None if window is None else float(window)


def window_label(window_s: float) -> str:
    """How a teacher window reads in a run label or an error: ``first 5 min``, ``first 90 s``."""
    if window_s >= 60 and float(window_s) % 60 == 0:
        return f"first {int(window_s // 60)} min"
    return f"first {window_s:g} s"


def refuse_teacher_run(record: dict | None, *, consumer: str, upstream: str) -> None:
    """Raise when ``record`` is a teacher run and ``consumer`` needs the whole recording.

    Args:
        record: The upstream run's record (a ``*_runs`` entry, or ``keypoints.yaml``).
        consumer: What is asking, as a user would name it ("embedding", "location analysis").
        upstream: How to name the upstream run in the message (``"tracking run trk_002"``).

    Raises:
        ValueError: The run covers only its recording's first minutes.
    """
    window = window_of(record)
    if window is None:
        return
    raise ValueError(
        f"{upstream} is a keypoint teacher run ({window_label(window)} of the recording); "
        f"{consumer} needs a run over the whole recording. Run detection without a "
        "teacher window and use that run instead."
    )
