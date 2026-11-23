"""Trajectory continuity for clip work (§16) — the helpers every clip consumer shares.

These work on one **subject**'s in-memory ``(embedding, frame_idx, t_ns)`` arrays (a subject
is a ``(recording_id, track_id)``).

**Continuity.** Embedding rows exist only where the subject was detected + tracked, so a
coast, a miss, or a part boundary leaves a hole. Matching or pooling a trajectory across such
a hole is meaningless (the path premise breaks), so a subject's trajectory is split into
**continuous runs** wherever consecutive ``t_ns`` jump by more than
``trajectory_split_gap_ns`` (default a small multiple of the subject's own frame period):
``segment_runs`` makes the split, and ``span_to_rows`` / ``frame_span_to_t_ns`` /
``band_rows`` convert between a span's time, frame and row spellings. ``check_span`` returns
"does this clip lie inside one run" as a **verdict** rather than an exception, so a surface
that picks examples can say what is wrong at the point of picking; ``run_frame_ranges`` gives
it the continuity to draw.

The cross-subject search that sat on top of these (``query_tracks``, over the windowed
matchers ``retrieval.py`` then held) was removed with query eval, its last caller.

Pure over arrays (Tenet 2/3 — no zarr, no CSV, no paths); the module and the CLI/GUI own
the I/O.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# max_gap default = this many nominal frame periods (a 1-2 frame miss stays in one run;
# a real gap splits). Tunable per call via ``trajectory_split_gap_ns``.
DEFAULT_GAP_FACTOR = 4.0


@dataclass(frozen=True)
class Gap:
    """A break in a subject's trajectory — two consecutive embedded rows further apart than
    the split threshold, i.e. a stretch where the subject went undetected."""

    start_t_ns: int
    end_t_ns: int
    start_frame: int
    end_frame: int

    @property
    def duration_ns(self) -> int:
        return self.end_t_ns - self.start_t_ns

    @property
    def n_frames(self) -> int:
        """Frames between the two rows — the stretch carrying no detection."""
        return max(self.end_frame - self.start_frame - 1, 0)


@dataclass(frozen=True)
class SpanCheck:
    """What a marked time span resolves to on one subject's trajectory, and why it is not
    usable as a query example if it isn't. The span fields are the **snapped** span — the
    rows actually covered, which a mark on an undetected frame pulls inward.

    ``gaps`` is populated whether or not they made the span unusable, so a surface can *say*
    a clip crosses a break while still accepting it — the case a method that pools exemplar
    frames into an unordered index (``soft_knn``) is in.
    """

    ok: bool
    reason: str | None = None
    start_row: int | None = None
    end_row: int | None = None
    start_t_ns: int | None = None
    end_t_ns: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    gaps: tuple[Gap, ...] = ()

    @property
    def n_rows(self) -> int:
        if self.start_row is None or self.end_row is None:
            return 0
        return self.end_row - self.start_row + 1

    @property
    def gap_note(self) -> str | None:
        """The breaks this span crosses, described — whether or not they refused it.

        One formatter for one sentence: ``reason`` says why a span was refused and is
        ``None`` when it was not, so a surface that accepts gap-crossing clips and still
        wants to describe them would otherwise grow a second, drifting copy of this text.
        """
        return _gap_reason(self.gaps) if self.gaps else None


def nominal_period_ns(t_ns: np.ndarray) -> int:
    """Median positive step between consecutive embedded frames (0 if < 2 rows) — the
    subject's cadence, robust to a minority of gap outliers."""
    if len(t_ns) < 2:
        return 0
    diffs = np.diff(np.asarray(t_ns, dtype=np.int64))
    positive = diffs[diffs > 0]
    return int(np.median(positive)) if positive.size else 0


def default_max_gap_ns(t_ns: np.ndarray) -> int:
    """``DEFAULT_GAP_FACTOR`` × the cadence (0 if the cadence is unknown → no split)."""
    return int(DEFAULT_GAP_FACTOR * nominal_period_ns(t_ns))


def segment_runs(t_ns: np.ndarray, max_gap_ns: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` row ranges of continuous runs — split wherever a
    consecutive ``t_ns`` step exceeds ``max_gap_ns``. ``max_gap_ns <= 0`` → one run."""
    n = len(t_ns)
    if n == 0:
        return []
    if max_gap_ns <= 0:
        return [(0, n)]
    diffs = np.diff(np.asarray(t_ns, dtype=np.int64))
    breaks = np.nonzero(diffs > max_gap_ns)[0]  # a gap sits between row b and b+1
    bounds = [0, *(int(b) + 1 for b in breaks), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def span_to_rows(
    row_t_ns: np.ndarray, start_t_ns: int, end_t_ns: int
) -> tuple[int, int]:
    """Inclusive embedding-row span covering the ``t_ns`` time range (Tenet 8).

    Rows carry ``t_ns`` in ascending order, so the span resolves by ``searchsorted`` —
    time is the reference (robust to downsampling / cadence), frames are display only.

    Raises
    ValueError: no embedded rows fall in ``[start_t_ns, end_t_ns]``
    """
    if end_t_ns < start_t_ns:
        raise ValueError(f"end_t_ns {end_t_ns} is before start_t_ns {start_t_ns}")
    lo = int(np.searchsorted(row_t_ns, start_t_ns, side="left"))
    hi = int(np.searchsorted(row_t_ns, end_t_ns, side="right")) - 1
    if hi < lo:
        raise ValueError(
            f"no embedded rows in time range [{start_t_ns}, {end_t_ns}] ns"
        )
    return lo, hi


def frame_span_to_t_ns(
    frame_idx: np.ndarray, row_t_ns: np.ndarray, start_frame: int, end_frame: int
) -> tuple[int, int]:
    """Map a global-frame span to the ``t_ns`` span of the embedded rows it covers — the
    frame-input convenience (CLI) that normalizes to the ``t_ns`` reference (Tenet 8).

    ``frame_idx`` must be counted across the whole recording, as the timeline loaders return
    it: a binary search over part-local numbers, which restart in each video part, silently
    lands in whichever part the search happens to reach (issue #23), so an array that is not
    strictly increasing is refused rather than searched.

    Raises
    ValueError: no embedded rows fall in ``[start_frame, end_frame]``, or ``frame_idx`` is
    not strictly increasing
    """
    if end_frame < start_frame:
        raise ValueError(f"end_frame {end_frame} is before start_frame {start_frame}")
    frame_idx = np.asarray(frame_idx)
    if len(frame_idx) > 1 and not bool(np.all(np.diff(frame_idx) > 0)):
        raise ValueError(
            "frame_idx is not strictly increasing, so it is not one recording's frame "
            "numbers (part-local frames restart in each video part); load the timeline "
            "with run_preview.load_track_timeline"
        )
    lo = int(np.searchsorted(frame_idx, start_frame, side="left"))
    hi = int(np.searchsorted(frame_idx, end_frame, side="right")) - 1
    if hi < lo:
        raise ValueError(
            f"no embedded rows in frame range [{start_frame}, {end_frame}]"
        )
    return int(row_t_ns[lo]), int(row_t_ns[hi])


def band_rows(subject_t_ns: np.ndarray, band_ns: int) -> int:
    """Convert a warp tolerance in **nanoseconds** to database rows for one subject.

    A given ± seconds is a different number of rows per cadence, so the conversion is
    per-subject: ``round(band_ns / nominal_period_ns)``. Returns 0 (rigid) when the
    cadence is unknown or ``band_ns <= 0``."""
    period = nominal_period_ns(subject_t_ns)
    if period <= 0 or band_ns <= 0:
        return 0
    return int(round(band_ns / period))


def _gap_for(t_ns: np.ndarray, max_gap_ns: int | None) -> int:
    return default_max_gap_ns(t_ns) if max_gap_ns is None else int(max_gap_ns)


def run_frame_ranges(
    frame_idx: np.ndarray, t_ns: np.ndarray, max_gap_ns: int | None = None
) -> list[tuple[int, int]]:
    """Each continuous run as an inclusive ``(first_frame, last_frame)`` global-frame range.

    ``frame_idx`` is the recording-wide numbering the timeline loaders return, so a track
    continuous over two video parts is one run, not one per part.

    ``segment_runs`` in the frame numbers a clip picker scrubs over, so a surface can draw
    the same continuity the example rule enforces.
    """
    runs = segment_runs(t_ns, _gap_for(t_ns, max_gap_ns))
    return [(int(frame_idx[a]), int(frame_idx[b - 1])) for a, b in runs]


def _gaps_in(
    frame_idx: np.ndarray, t_ns: np.ndarray, lo: int, hi: int, max_gap_ns: int
) -> tuple[Gap, ...]:
    """The splitting breaks strictly inside rows ``[lo, hi]``."""
    if max_gap_ns <= 0 or hi <= lo:
        return ()
    t = np.asarray(t_ns, dtype=np.int64)
    breaks = np.nonzero(np.diff(t[lo : hi + 1]) > max_gap_ns)[0] + lo
    return tuple(
        Gap(int(t[b]), int(t[b + 1]), int(frame_idx[b]), int(frame_idx[b + 1]))
        for b in breaks.tolist()
    )


def _gap_reason(gaps: Sequence[Gap]) -> str:
    """The one rendering of "this span crosses a break", for every surface that says so.

    The frame count leads because it is exact and never zero; the duration joins it only
    when it is a number a person can act on. Two decimal places round a real break at a high
    embed cadence to ``0.00 s``, which reads as no break at all — and this sentence is now
    shown beside *accepted* clips, not only in a rare refusal, so it is read far more often.
    """
    longest = max(gaps, key=lambda g: g.duration_ns)
    frames = longest.n_frames
    detail = f"{frames} frame{'' if frames == 1 else 's'} with no detection"
    seconds = longest.duration_ns / 1e9
    if seconds >= 0.01:
        detail = f"{seconds:.2f} s ({detail})"
    if len(gaps) == 1:
        return f"crosses a break in tracking of {detail}"
    return f"crosses {len(gaps)} breaks in tracking; the longest is {detail}"


def check_span(
    frame_idx: np.ndarray,
    t_ns: np.ndarray,
    start_t_ns: int,
    end_t_ns: int,
    max_gap_ns: int | None = None,
    *,
    require_contiguous: bool = True,
) -> SpanCheck:
    """Whether a time span is usable as a query example, and why not if it isn't.

    The rule ``_resolve_example`` enforces, returned as a verdict instead of raised as an
    exception — so every surface that *picks* an example (the clip picker, the CSV checker,
    the CLI) can say what is wrong at the point of picking rather than at search time.

    ``require_contiguous`` is the asking method's answer, never this module's guess
    (``ExemplarBounds.requires_contiguous``). With it ``False`` a gap-crossing span is
    **ok** and its ``gaps`` are still reported, so the picker can describe the break instead
    of refusing it. The two checks above it are unconditional: a span that resolves to no
    embedded rows at all is unusable to every method, because there is nothing to pool.
    """
    if len(t_ns) == 0:
        return SpanCheck(
            ok=False, reason="the track has no embedded frames in this run"
        )
    if end_t_ns < start_t_ns:
        return SpanCheck(ok=False, reason="the span ends before it starts")
    try:
        lo, hi = span_to_rows(t_ns, start_t_ns, end_t_ns)
    except ValueError:
        return SpanCheck(
            ok=False,
            reason="has no embedded frames — the subject was not detected anywhere in it",
        )
    gaps = _gaps_in(frame_idx, t_ns, lo, hi, _gap_for(t_ns, max_gap_ns))
    refused = bool(gaps) and require_contiguous
    return SpanCheck(
        ok=not refused,
        reason=_gap_reason(gaps) if refused else None,
        start_row=lo,
        end_row=hi,
        start_t_ns=int(t_ns[lo]),
        end_t_ns=int(t_ns[hi]),
        start_frame=int(frame_idx[lo]),
        end_frame=int(frame_idx[hi]),
        gaps=gaps,
    )
