"""Time windows over a recording (§23).

A window is a half-open ``[start_t_ns, end_t_ns)`` interval, and **the whole recording is simply
the window that covers everything** — so there is no separate per-recording code path anywhere
above this module. Everything a measure answers is a windowed question, and because the artifact
stores intervals the answer is exact: no binning, ever. Binning is a choice of *where to evaluate*,
not a property of the data, and it never changes what an answer is.

**Half-open, and the reason is additivity.** With both bounds inclusive a frame on a shared
boundary falls in *both* neighbouring windows and is credited a full period to each, so two halves
of a recording sum to more than the whole and a window one period long reports twice its own
length. §23's additivity guard — a measure must sum across adjacent windows to the value over their
union — is not merely failing under inclusive bounds, it is unpassable. ``frame_view.observed_ns_in``
already takes this convention; this module is where windows are built to match it.

Four builders, one shape:

- **whole** — the coverage span, the default.
- **absolute** — fixed durations from the recording's start, for cohorts of like length and for
  event-referenced work later.
- **normalized** — equal fractions of each recording, for cohorts whose recordings differ in
  length. Comparable *positions* rather than comparable durations; the two answer different
  questions and both are wanted.
- **sliding** — a window stepped along the recording, which is how a measure becomes a curve.
  "Rearing rate across the session" is the same measure evaluated at many windows, not a different
  kind of measure.

Pure and dependency-light on purpose: §24 slices by time bin through these same builders, so a
window is not a measures concept.
"""

from dataclasses import dataclass


class WindowError(ValueError):
    """A window specification that cannot produce a usable window."""


@dataclass(frozen=True)
class Window:
    """A half-open ``[start_t_ns, end_t_ns)`` span with an id that names it in output.

    ``window_id`` is what a reader groups by, so it is stable and descriptive rather than an
    index: ``whole``, ``abs_00``, ``third_02``. An index alone stops meaning anything the moment
    two runs use different window settings.
    """

    window_id: str
    start_t_ns: int
    end_t_ns: int

    def __post_init__(self) -> None:
        if self.end_t_ns < self.start_t_ns:
            raise WindowError(
                f"window {self.window_id!r} ends before it starts "
                f"({self.start_t_ns} > {self.end_t_ns})"
            )

    @property
    def length_ns(self) -> int:
        """Wall-clock span. **Not** a denominator — see §23; observed time is what divides."""
        return self.end_t_ns - self.start_t_ns

    def contains_instant(self, t_ns: int) -> bool:
        """Whether ``t_ns`` falls in this window, half-open."""
        return self.start_t_ns <= t_ns < self.end_t_ns

    def overlap_ns(self, start_t_ns: int, end_t_ns: int) -> int:
        """Overlap of the half-open span ``[start, end)`` with this window, in ns.

        Callers pass an instance's *measured* span — ``[t_ns, end_t_ns + period)``, not the
        stored inclusive one — so that a single-frame bout contributes one period rather than
        zero (§23 "One convention: half-open").
        """
        lo = max(self.start_t_ns, int(start_t_ns))
        hi = min(self.end_t_ns, int(end_t_ns))
        return max(hi - lo, 0)


def window_sort_key(window_id: str) -> tuple[int, str]:
    """Order window ids the way the windows themselves run.

    Every builder here mints a zero-padded counter (``abs_07``, ``part_02``, ``slide_014``), and
    the padding is a *minimum* width — the 100th absolute window is ``abs_100``. Plain
    lexicographic order then reads ``abs_09, abs_10, abs_100, abs_101, ..., abs_11``, because
    ``'0' < '1'`` at the third digit. A curve plotted in that order is a scribble, and its x-axis
    labels are in that order too, so nothing about the figure says it happened.

    Reachable at ordinary settings: a 30-minute recording windowed every 10 seconds is 180
    windows. Sorting on the id's length first restores the numeric order for every id these
    builders produce, without teaching this module to parse its own names.
    """
    return (len(window_id), window_id)


def whole(start_t_ns: int, end_t_ns: int) -> list[Window]:
    """The one window covering everything — the default, and not a special case anywhere above."""
    return [Window("whole", int(start_t_ns), int(end_t_ns))]


def absolute(start_t_ns: int, end_t_ns: int, *, every_ns: int) -> list[Window]:
    """Fixed-duration windows tiling ``[start, end)``.

    The final window is **short rather than dropped or over-long**: dropping it loses real time
    from the tail, and extending it past the recording gives the last window a denominator no
    observation can fill. A short last window is honest and its own denominator says so.
    """
    if every_ns <= 0:
        raise WindowError(f"every_ns must be > 0, got {every_ns}")
    start, end = int(start_t_ns), int(end_t_ns)
    out: list[Window] = []
    edge = start
    while edge < end:
        out.append(Window(f"abs_{len(out):02d}", edge, min(edge + every_ns, end)))
        edge += every_ns
    return out


def normalized(start_t_ns: int, end_t_ns: int, *, parts: int) -> list[Window]:
    """``parts`` equal fractions of the recording — comparable positions, not durations.

    The last edge is pinned to ``end`` rather than accumulated, so integer division cannot leave
    a sliver of the recording in no window at all.
    """
    if parts <= 0:
        raise WindowError(f"parts must be > 0, got {parts}")
    start, end = int(start_t_ns), int(end_t_ns)
    span = end - start
    out: list[Window] = []
    for i in range(parts):
        lo = start + (span * i) // parts
        hi = end if i == parts - 1 else start + (span * (i + 1)) // parts
        out.append(Window(f"part_{i:02d}", lo, hi))
    return out


def sliding(
    start_t_ns: int, end_t_ns: int, *, length_ns: int, step_ns: int
) -> list[Window]:
    """A window of ``length_ns`` stepped by ``step_ns`` — how a measure becomes a curve.

    Windows may overlap (``step < length``), which is the point: a smooth curve wants overlap.
    Overlapping windows are **not additive**, and nothing here pretends otherwise — additivity
    is asserted over *adjacent* windows, which is what ``absolute`` and ``normalized`` produce.
    """
    if length_ns <= 0:
        raise WindowError(f"length_ns must be > 0, got {length_ns}")
    if step_ns <= 0:
        raise WindowError(f"step_ns must be > 0, got {step_ns}")
    start, end = int(start_t_ns), int(end_t_ns)
    out: list[Window] = []
    edge = start
    while edge < end:
        out.append(Window(f"slide_{len(out):03d}", edge, min(edge + length_ns, end)))
        edge += step_ns
    return out
