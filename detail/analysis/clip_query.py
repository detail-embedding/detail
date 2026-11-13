"""Clip-query orchestration (§16) — retrieval over loaded pose-embedding trajectories,
on top of the banded-DTW matcher in ``retrieval.py``.

Given in-memory ``(embedding, frame_idx, t_ns)`` arrays per **subject** (a detail
``(recording_id, track_id)`` — the earlier prototype's "recording") and a set of example clips, this
enforces temporal continuity, fuses the examples, searches each requested subject, and
globally ranks the matches.

**Continuity.** Embedding rows exist only where the subject was detected + tracked, so a
coast, a miss, or a part boundary leaves a hole. Matching a query trajectory across such
a hole is meaningless (the path premise breaks), so each subject's trajectory is first
split into **continuous runs** wherever consecutive ``t_ns`` jump by more than
``trajectory_split_gap_ns`` (default a small multiple of the subject's own frame
period). A query
example must lie inside one run; the DTW searches each run independently, so a match can
never straddle a gap. ``check_span`` returns that same rule as a **verdict** rather than an
exception, so a surface that picks examples can say what is wrong at the point of picking;
``run_frame_ranges`` gives it the continuity to draw.

Pure over arrays (Tenet 2/3 — no zarr, no CSV, no paths); the module and the CLI/GUI own
the I/O.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from detail.analysis import retrieval
from detail.core.utils import coerce_series_key

# A subject key: (recording_id, track_id). One continuous-run search unit per (subject, run).
Subject = tuple[str, int]

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


@dataclass(frozen=True)
class Example:
    """A resolved query example: its subject + inclusive row span + frame/t_ns span."""

    recording_id: str
    track_id: int
    start_row: int
    end_row: int
    start_frame: int
    end_frame: int
    start_t_ns: int
    end_t_ns: int


@dataclass(frozen=True)
class RankedMatch:
    """A retrieved stretch: subject + inclusive row span + frame/t_ns span + score + the
    index of the example clip it best matches (0 for a single-example query)."""

    recording_id: str
    track_id: int
    start_row: int
    end_row: int
    start_frame: int
    end_frame: int
    start_t_ns: int
    end_t_ns: int
    score: float
    example_idx: int = 0


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

    Raises
    ValueError: no embedded rows fall in ``[start_frame, end_frame]``
    """
    if end_frame < start_frame:
        raise ValueError(f"end_frame {end_frame} is before start_frame {start_frame}")
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


def _resolve_example(
    databases: dict[Subject, tuple[np.ndarray, np.ndarray, np.ndarray]],
    spec: dict,
    max_gap_ns: int | None,
) -> tuple[Example, np.ndarray]:
    """Resolve one ``{recording_id, track_id, start_t_ns, end_t_ns}`` spec to an
    ``Example`` (with its embeddings), validating it lies within one continuous run."""
    subject = (spec["recording_id"], coerce_series_key(spec["track_id"]))
    if subject not in databases:
        raise KeyError(f"query subject {subject!r} not loaded")
    emb, frame_idx, t_ns = databases[subject]
    start_t_ns, end_t_ns = int(spec["start_t_ns"]), int(spec["end_t_ns"])
    check = check_span(frame_idx, t_ns, start_t_ns, end_t_ns, max_gap_ns)
    if not check.ok:
        raise ValueError(
            f"example clip {subject[0]!r} track {subject[1]} "
            f"{start_t_ns / 1e9:.2f}–{end_t_ns / 1e9:.2f} s {check.reason}; pick an "
            "example within one continuous stretch"
        )
    lo, hi = check.start_row, check.end_row
    example = Example(
        recording_id=subject[0],
        track_id=subject[1],
        start_row=lo,
        end_row=hi,
        start_frame=int(frame_idx[lo]),
        end_frame=int(frame_idx[hi]),
        start_t_ns=int(t_ns[lo]),
        end_t_ns=int(t_ns[hi]),
    )
    return example, emb[lo : hi + 1]


# Nominal share of one subject's work spent loading its trajectory before any DTW runs.
# A bar may use nominal phase weights (§6); an ETA may not, which is why this scales the
# bar only and the runner derives its ETA from the folded fraction.
_LOAD_SHARE = 0.2


def query_tracks(
    databases: dict[Subject, tuple[np.ndarray, np.ndarray, np.ndarray]],
    queries: Sequence[dict],
    search: Sequence[Subject],
    *,
    fusion: str = "mean",
    band_ns: int,
    threshold: float,
    min_separation: int | None = None,
    trajectory_split_gap_ns: int | None = None,
    max_results: int | None = None,
    overlap_ns: int | None = None,
    method: str = "dtw",
    method_params: dict | None = None,
    labeled: Sequence[dict] | None = None,
    progress_cb: Callable[[int, int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> tuple[list[Example], list[RankedMatch]]:
    """Stretches across ``search`` subjects within ``threshold`` of the fused example
    ``queries``, continuity-segmented and globally ranked.

    Args
    databases: (recording_id, track_id) -> (emb (M, D), frame_idx (M,), t_ns (M,)); must
        include every subject referenced by ``search`` and ``queries``
    queries: example clips ``[{recording_id, track_id, start_t_ns, end_t_ns}, ...]``
    search: the subjects to search over (an example may come from a subject not searched)
    fusion: ``"mean"`` (consensus) or ``"min"`` (any) — see ``retrieval.query_multi``
    band_ns: warp / duration tolerance in **nanoseconds** (0 = rigid fixed-length),
        converted to rows per subject (Tenet 8 — time is the reference)
    threshold: keep matches whose fused normalized cosine distance is <= this
    min_separation: min row gap between matches; ``None`` = the longest example length
        (ignored when ``overlap_ns`` is set — see ``retrieval.query_multi``)
    trajectory_split_gap_ns: a break in tracking longer than this SPLITS the recording
        into separate stretches to search; ``None`` = from the subject's own frame rate
    max_results: optional global safety cap on the pooled result (logged if it truncates)
    overlap_ns: if set (nanoseconds, converted to rows per subject), matches and excluded
        spans may overlap by up to this much instead of being kept fully apart — the
        annotation loop's tolerance (``None`` = clip-query's strict non-overlap)
    labeled: already-labeled spans ``[{recording_id, track_id, start_t_ns, end_t_ns}, ...]``
        to exclude from results (the annotation loop; merged into each subject's excludes)
    progress_cb: called ``(done, subjects_total, matches_so_far, phase)``. ``done`` is
        **fractional** — the whole part is subjects finished, the remainder is progress
        inside the running subject — and ``phase`` names what it is doing right now
        (``loading <recording>`` / ``searching <recording> · stretch i/n``). A subject is
        a lazy trajectory read plus a DTW pass per continuous stretch, both slow; emitting
        only at the subject boundary leaves a one-recording search pinned at 0 for the
        whole run and then jumping to 100.
    stop_check: polled before each subject; a truthy return ends the search early

    Returns
    examples: each query's resolved subject + spans (for the CSV query rows)
    matches: RankedMatch, best (lowest score) first
    """
    if not queries:
        raise ValueError("queries is empty")
    if not search:
        raise ValueError("no subjects to search")

    examples: list[Example] = []
    query_embs: list[np.ndarray] = []
    for spec in queries:
        example, emb = _resolve_example(databases, spec, trajectory_split_gap_ns)
        examples.append(example)
        query_embs.append(emb)

    longest = max(len(q) for q in query_embs)
    min_sep = int(min_separation) if min_separation is not None else longest

    pooled: list[RankedMatch] = []
    total = len(search)

    def report(done: float, phase: str) -> None:
        if progress_cb is not None:
            progress_cb(done, total, len(pooled), phase)

    for done, subject in enumerate(search, start=1):
        if stop_check is not None and stop_check():
            break
        if subject not in databases:
            raise KeyError(f"search subject {subject!r} not loaded")
        # The trajectory read is lazy (one subject at a time) and is the slow half on a
        # long recording, so it is named before it starts rather than after it ends, and
        # holds a nominal share of the subject — otherwise a subject with one continuous
        # stretch has only two states and the bar still jumps.
        report(done - 1, f"loading {subject[0]}")
        emb, frame_idx, t_ns = databases[subject]
        runs = segment_runs(t_ns, _gap_for(t_ns, trajectory_split_gap_ns))
        subject_band = band_rows(t_ns, band_ns)  # ± ns → rows at this subject's cadence
        # None = clip-query's strict non-overlap; a value (rows) = the annotation tolerance.
        subject_overlap = None if overlap_ns is None else band_rows(t_ns, overlap_ns)
        subject_excludes = [
            (e.start_row, e.end_row)
            for e in examples
            if (e.recording_id, e.track_id) == subject
        ]
        if labeled is not None:
            for lab in labeled:
                if (
                    lab["recording_id"],
                    coerce_series_key(lab["track_id"]),
                ) != subject:
                    continue
                lo, hi = span_to_rows(
                    t_ns, int(lab["start_t_ns"]), int(lab["end_t_ns"])
                )
                subject_excludes.append((lo, hi))
        n_runs = max(len(runs), 1)
        for run_i, (run_start, run_end) in enumerate(runs):
            # Within-subject progress: the DTW pass over each stretch is the other slow
            # half, and there can be hundreds of them on one recording.
            report(
                done - 1 + _LOAD_SHARE + (1 - _LOAD_SHARE) * (run_i / n_runs),
                f"searching {subject[0]} · stretch {run_i + 1}/{n_runs}",
            )
            if run_end - run_start < longest:  # too short for a full-query match
                continue
            excludes = [
                (max(a, run_start) - run_start, min(b, run_end - 1) - run_start)
                for a, b in subject_excludes
                if b >= run_start and a < run_end
            ]
            matches = retrieval.query_multi(
                emb[run_start:run_end],
                query_embs,
                fusion=fusion,
                band=subject_band,
                threshold=threshold,
                min_separation=min_sep,
                excludes=excludes,
                overlap=subject_overlap,
                method=method,
                method_params=method_params,
            )
            for m in matches:
                g0, g1 = m.start_row + run_start, m.end_row + run_start
                pooled.append(
                    RankedMatch(
                        recording_id=subject[0],
                        track_id=subject[1],
                        start_row=g0,
                        end_row=g1,
                        start_frame=int(frame_idx[g0]),
                        end_frame=int(frame_idx[g1]),
                        start_t_ns=int(t_ns[g0]),
                        end_t_ns=int(t_ns[g1]),
                        score=m.score,
                        example_idx=m.example_idx,
                    )
                )
        # A `done` head closes this subject's pass through the phases (core.progress.message_phase).
        report(done, f"done · {subject[0]}")

    pooled.sort(key=lambda m: m.score)
    if max_results is not None and len(pooled) > max_results:
        retrieval._log.warning(
            "clip query pooled %d matches over threshold=%g; returning the top "
            "max_results=%d — raise max_results or lower the threshold to see the rest",
            len(pooled),
            threshold,
            max_results,
        )
        pooled = pooled[:max_results]
    return examples, pooled
