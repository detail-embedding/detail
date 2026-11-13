"""Shared progress-emission cadence for long-running runners (§6).

Every long-running runner (detection, tracking, embedding, dataset
export) must emit progress *incrementally* so that even a single-part / single-unit
run shows a live bar — not one event at the end. The recurring bug is a runner
that only emits at coarse boundaries (per part / per unit); this module gives the
synchronous, in-loop emitters one shared throttle so the cadence is defined in a
single place and a new stage can reuse it instead of hand-rolling (and forgetting)
its own.

Multiprocess coordinators (detection / embedding) emit by polling their worker
progress files every ``PROGRESS_INTERVAL_S`` and so are already incremental; they
share the interval constant here but not the wrapper, whose ``last_emit`` shape
doesn't fit a ``worker.join(timeout)`` loop.
"""

import functools
import logging
import time
from typing import Callable, Iterable, Iterator, TypeVar

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

PROGRESS_INTERVAL_S = 0.5  # the single live-bar cadence for every runner

#: The two passes over every selected recording before a processing run is launched (and
#: before a retry re-launches one): the stage names ``Experiment.run_processing`` reports under.
VALIDATING_STAGE = "Validating recordings"
ESTIMATING_STAGE = "Estimating disk space"
#: A module's ``resolve_shared_config`` pass, before validation, when it reads recordings
#: (embedding's shared canvas); a module that resolves nothing shared reports no such stage.
RESOLVING_STAGE = "Resolving shared settings"

MESSAGE_SEPARATOR = " · "  # between a message's phase head and its note
DONE_PHASE = (
    "done"  # the head of a runner's terminal message — every phase is behind it
)


class OperationCancelled(Exception):
    """Raised **from a progress callback** when the person watching asked to stop.

    A synchronous operation that reports ``(stage, fraction, detail)`` is cancellable at its
    next report without carrying a stop flag of its own: the GUI's ``busy.run_progress`` raises
    this from the callback once Cancel is pressed or its window closes. Code that catches broad
    exceptions inside such an operation must let it through; code that writes as it goes must
    leave nothing half-made behind it (``subject_frames.sample_frames`` removes the images a
    cancelled draw created).
    """


def message_phase(message: str, phases: tuple[str, ...]) -> int | None:
    """Which of `phases` (an ``AnalysisModule.phases`` declaration) `message` names, or ``None``.

    A runner's message is ``"<phase head> · <note>"``, and only the **head** is read. The note is
    the runner's to fill with anything — a user's behavior or recording name, or an inner step
    such as the ``loading <subject>`` that query-eval's querying phase forwards from
    ``query_tracks`` — so a phase word appearing there says nothing about the phase. Matching the
    whole message put every query-eval run back on "loading" once per subject it searched.

    The head's first word is the phase's first word, and the phase's other words are in the head
    too, so ``"comparing 4,182 bouts"`` is the phase ``"comparing bouts"`` (the count is the
    reason the message exists). Returns ``len(phases)`` for a ``done`` head: every phase is over.
    """
    head = message.split(MESSAGE_SEPARATOR, 1)[0].split()
    if not head:
        return None
    if head[0] == DONE_PHASE:
        return len(phases)
    for i, phase in enumerate(phases):
        words = phase.split()
        if words and words[0] == head[0] and set(words) <= set(head):
            return i
    return None


def best_effort(what: str):
    """Decorate a **progress-publishing** function so it can never raise.

    Progress is a statement *about* work, not the work. Losing one sample is invisible — the
    next one overwrites it — while letting the write fail is not: in #289 a transient Windows
    ``PermissionError`` on the coordinator's own ``progress.json`` rename was caught by the
    handler meant for *job* failures, so a healthy recording was recorded ``failed``, its worker
    was orphaned mid-run (nothing waited on it, and it went on competing for the GPU), and its
    ``recording.yaml`` never received a terminal status.

    **Retrying is already in place and is not the answer.** ``core.atomic`` retries ``os.replace``
    twenty times at 10ms and the error still escaped; ``test_atomic`` even asserts that a bounded
    retry re-raises once exhausted. What was missing is a contract for *what the caller does
    then*. This is that contract: for progress, the answer is "nothing".

    Apply it **only to the persistence of progress** — never to a status write (a source of
    truth, which must raise) and never to a caller-supplied callback, whose exceptions are its
    own bugs and belong to whoever wrote it. ``rundir.set_progress`` and ``runs.write_progress``
    are the two functions this covers; their siblings ``set_status`` / ``set_pid`` /
    ``update_payload`` share a writer and deliberately do not.

    The failure is logged at WARNING rather than swallowed silently: a share that can *never* be
    written leaves a bar frozen, and the log is the only thing that says why.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def published(*args, **kwargs):
            # noqa: BLE001 — the whole point of this wrapper: publishing cannot fail.
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "could not publish %s (%s: %s) — the run is unaffected",
                    what,
                    type(exc).__name__,
                    exc,
                )
                return None

        #: The raising original, so a test of the writer itself can still assert it raises.
        published.unpublished = fn
        return published

    return decorate


class ThrottledProgress:
    """Wrap a ``progress_cb`` so a runner can emit on every iteration cheaply.

    Forwards an event to the wrapped callback only when ``force=True``, on the
    first call, or once ``interval_s`` has elapsed since the last forward. The
    first-call rule makes a single-part / single-unit run show a bar immediately;
    pass ``force=True`` on the terminal event so the final 100% event is never
    dropped by the throttle.

    ``interval_s=None`` (the default) reads the module-level ``PROGRESS_INTERVAL_S``
    at call time, so a test can ``monkeypatch`` that one constant to ``0.0`` and
    make every runner emit per-iteration deterministically.
    """

    def __init__(
        self, progress_cb: Callable | None, interval_s: float | None = None
    ) -> None:
        self._cb = progress_cb
        self._interval_s = interval_s
        self._last: float | None = None

    def __call__(self, event, *, force: bool = False) -> None:
        if self._cb is None:
            return
        interval = (
            self._interval_s if self._interval_s is not None else PROGRESS_INTERVAL_S
        )
        now = time.monotonic()
        if force or self._last is None or now - self._last >= interval:
            self._cb(event)
            self._last = now


WITHIN_RESOLUTION = 1000  # inner-fraction denominator for ``split_units``


def split_units(done: float, total: int) -> tuple[int, int, int]:
    """Split a fractional unit count into an analysis event's inner/outer progress.

    ``gui/workers.py::fit_progress_fraction`` folds an analysis event as
    ``(recordings_done + frames_done / frames_total) / recordings_total``, so a runner
    whose real progress is fractional reports the whole part as the outer count and the
    remainder through the inner pair. A runner that reports only whole units emits one
    event at 0 and one at the end: on a single-unit run the bar sits dead and then jumps
    to 100, which reads as a hang. Returns ``(outer_done, inner_done, inner_total)``.
    """
    outer = min(int(done), max(total - 1, 0))
    within = min(max(done - outer, 0.0), 1.0)
    return outer, round(within * WITHIN_RESOLUTION), WITHIN_RESOLUTION


def remaining_seconds(fraction: float, elapsed_s: float) -> float | None:
    """Estimate seconds remaining from the completed ``fraction`` (0..1) and the
    time spent so far: ``elapsed_s * (1 - fraction) / fraction`` — the linear
    estimate the embedding coordinator uses. Returns ``None`` when it can't be
    computed (no progress yet, already complete, or no elapsed time)."""
    if 0 < fraction < 1 and elapsed_s > 0:
        return elapsed_s * (1 - fraction) / fraction
    return None


class PhaseClock:
    """An ETA measured over the *running phase*, not the whole run.

    A multi-phase runner folds its phases into one bar with nominal weights — labeling 45%
    of a distill, load 30% of the removed clusterer fit — because there is no common unit between
    frames and epochs, or between loading and DTW. Those weights are honest for a bar and
    wrong for a clock: ``remaining_seconds(f, e)`` assumes ``e`` bought all of ``f``, so
    feeding it a weighted fraction over a whole-run elapsed reads minutes on an hours-long
    run (PR #171).

    So the ETA divides the *phase's own* progress by the *phase's own* elapsed. The clock
    resets whenever the phase changes. For the last phase — training, DTW — the phase's
    remaining is the run's remaining, which is where an ETA matters most; earlier phases
    report their own remaining, since what follows them is not yet measurable.
    """

    def __init__(self, now: float):
        self._phase: str | None = None
        self._started = now

    def eta(self, phase: str, phase_fraction: float, now: float) -> float | None:
        """Seconds left in ``phase``, restarting the clock when the phase changes."""
        if phase != self._phase:
            self._phase = phase
            self._started = now
        return remaining_seconds(phase_fraction, now - self._started)


def reporting_each(
    items: Iterable[_T],
    stage: str,
    progress: Callable[[str, float, str], None] | None,
    *,
    label: Callable[[_T], str] | None = None,
) -> Iterator[_T]:
    """Yield each of ``items``, reporting ``(stage, fraction, detail)`` as it goes.

    The one way a synchronous loop over recordings tells its caller where it is: a report
    *before* each item (``detail`` is ``"<label> (N of M)"``, or ``"N of M"`` with no ``label``),
    and a final report at ``1.0`` once the last is done, so a one-item loop still shows a bar at
    both ends rather than none. Pre-run validation reads every recording's files, which on a
    network share or a 71-recording experiment is minutes — a status that never changes in that
    time reads as a hang.

    ``progress`` is the caller's own callback, so an exception from it is its own bug and is not
    caught. ``None`` reports nothing and costs nothing.
    """
    items = list(items)
    total = len(items)
    for index, item in enumerate(items):
        if progress is not None:
            count = f"{index + 1} of {total}"
            detail = f"{label(item)} ({count})" if label is not None else count
            progress(stage, index / total, detail)
        yield item
    if progress is not None and total:
        progress(stage, 1.0, f"{total} of {total}")


def format_duration(seconds: float) -> str:
    """Short human-readable duration, at most two units: ``45s`` / ``2m30s`` /
    ``1h05m``."""
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
