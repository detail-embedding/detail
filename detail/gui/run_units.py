"""What a run is made of, one row per part, with a bar each.

**The bug this replaces.** A run page had one bar for the whole run and, under it, a log. On a
detection run that meant a bar reading 0% for most of an hour above lines of
``22:12:40 · recording_A (1/2) · 1% · ~2h left`` — a scrolling terminal where a set
of bars belongs. The information was all there and none of it was legible: which recording is
going, how far, what failed, what is still waiting.

The log is not deleted, because it answers the other question — *has anything happened in the
last two hours* — which only a timestamped history can. It moves below these rows and folds.

**One shape, five units.** The part a run is made of is not the same thing twice: processing
loops **recordings**, analysis loops **phases**, distill loops **stages** (with a different
inner unit in each), training loops **epochs**, an export loops **targets**. Flattening them
into one stream of text is what made them all look alike. They share a row — name, bar, status
— and each surface says what its units are, which is why `LiveRunPanel.set_units` takes the
heading as a parameter rather than writing "recordings" into the widget.

**Grouping is not a display preference.** At a few dozen recordings, listing every one gives as many
full-width bars of which two are news: the one that is running and the one that failed. The
finished and the waiting are counts until asked for. Below `GROUP_ABOVE` everything lists,
because at three recordings a count is worse than the list it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from detail.core.progress import message_phase

#: Row states. `done` and `queued` are groupable — they are reference, and identical to each
#: other in the only way that matters. `running` and `failed` never group: they are the news.
DONE = "done"
RUNNING = "running"
QUEUED = "queued"
FAILED = "failed"

GROUPABLE = (DONE, QUEUED)

#: Every run status -> the row state it draws. **One table for every unit kind that reads a
#: status off a record** (a processing run's recordings, a distill's stages), so a status is
#: mapped once and a new one fails ``tests/gui/test_run_units.py`` until it is mapped here.
#:
#: The bug this replaces: each builder tested ``== "complete"`` and ``== "failed"`` and let
#: everything else fall through to *running* or *queued*. A cancelled recording read as
#: ``running · ~15s left`` under a live event and a partial one as ``queued`` — a stopped
#: recording described as still to come. ``cancelled`` and ``partial`` are over and did not
#: finish, so they draw as the news (`FAILED`, never grouped) with the status as their text.
#: ``pending`` / ``running`` are the two that defer to the live event: the record cannot know
#: which recording is in flight, so those draw `RUNNING` only for the one the event names.
STATUS_STATE = {
    "complete": DONE,
    "failed": FAILED,
    "cancelled": FAILED,
    "partial": FAILED,
    "running": RUNNING,
    "pending": QUEUED,
}

#: List every unit up to this many; beyond it, fold the groupable states into count rows.
#:
#: Eight because that is about where a list stops being read and starts being scrolled — and
#: because the runs this matters for are real cohorts, not the 2-recording tests.
#: The exact number is a judgement; what is not is that the threshold exists, since a widget
#: correct at the size it was written at and useless at the size the tool is *for* is this
#: package's recorded class.
GROUP_ABOVE = 8


@dataclass(frozen=True)
class Unit:
    """One row: a named part of a run, how far it got, and what to say about it.

    `fraction` is 0..1 and drives the bar; `status` is the right-hand text and is written by
    whoever built the unit, because what is worth saying differs by state — a finished recording
    reports what it produced and how long it took, a running one reports its phase and an ETA, a
    failed one reports why.
    """

    name: str
    state: str
    fraction: float
    status: str
    #: For a group row: the units it stands for, revealed when it is expanded.
    members: tuple["Unit", ...] = ()

    @property
    def is_group(self) -> bool:
        return bool(self.members)


def group_units(units: list[Unit], *, above: int = GROUP_ABOVE) -> list[Unit]:
    """Fold runs of groupable units into count rows, keeping order and keeping the news.

    Consecutive same-state runs collapse rather than all-of-a-state, so a cohort that failed in
    the middle still reads in the order it ran: seven done, the failure, the live one, twelve
    queued. Collapsing by state alone would move the failure out of its place in the sequence.
    """
    if len(units) <= above:
        return list(units)

    out: list[Unit] = []
    run: list[Unit] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            out.append(_group_of(run))
        run.clear()

    for unit in units:
        if unit.state in GROUPABLE and (not run or run[-1].state == unit.state):
            run.append(unit)
            continue
        flush()
        if unit.state in GROUPABLE:
            run.append(unit)
        else:
            out.append(unit)
    flush()
    return out


def _group_of(members: list[Unit]) -> Unit:
    state = members[0].state
    label = f"{len(members)} {state}"
    return Unit(
        name=label,
        state=state,
        fraction=1.0 if state == DONE else 0.0,
        status=_group_status(members, state),
        members=tuple(members),
    )


def _group_status(members: list[Unit], state: str) -> str:
    """What a count row says instead of the statuses it replaced.

    A range rather than a sum for finished work: "29m03s – 35m21s each" is the fact someone is
    looking for (are these consistent, is one an outlier), where a total answers a question
    nobody asked of a list they can already see the length of.
    """
    statuses = [u.status for u in members if u.status]
    if state == DONE and len(statuses) > 1:
        return f"{statuses[0]} – {statuses[-1]}"
    if state == DONE and statuses:
        return statuses[0]
    return ""


def processing_units(
    record: dict,
    progress=None,
    names: dict[str, str] | None = None,
) -> list[Unit]:
    """Rows for a processing run: one per recording, in the order the run will do them.

    Three sources, because no one of them knows everything:

    - ``recording_ids`` is the *plan* — including recordings the coordinator has not reached,
      which appear in nothing else and are the whole "what is still waiting" half.
    - ``per_recording`` is the *record*, written as each finishes, and is the only place a
      failure and its reason live. It survives the app closing, so a reopened page is complete.
    - the live ``ProcessingProgress`` names the recording in flight and how far into it, which
      the record cannot know until it is over.

    A recording in none of them is queued. Deriving this from the live event alone would lose
    every row on a page opened after the run started — which is the ordinary way of opening one.
    """
    names = names or {}
    per = record.get("per_recording") or {}
    ids = record.get("recording_ids") or list(per)
    current = getattr(progress, "current_recording", None)
    inner = getattr(progress, "inner", None)

    units = []
    for rid in ids:
        entry = per.get(rid) or {}
        status = entry.get("status")
        name = names.get(rid, rid)
        state = STATUS_STATE.get(status)
        if state == DONE:
            units.append(Unit(name, DONE, 1.0, entry.get("detail") or ""))
        elif state == FAILED:
            units.append(Unit(name, FAILED, 0.0, _ended(status, entry)))
        elif rid == current:
            units.append(Unit(name, RUNNING, _inner_fraction(inner), _running(inner)))
        else:
            units.append(Unit(name, QUEUED, 0.0, "queued"))
    return units


#: A stage's record key -> the phase name the live ``DistillProgress`` calls it, and the row
#: name it draws. In run order — a distill has no plan to enumerate the way a processing
#: run's ``recording_ids`` does: it is these stages, and a record draws the ones it lists
#: (a detector distill the first two, a keypoint distill all three — §12).
_DISTILL_STAGES = (
    ("label", "labeling", "Label frames"),
    ("train", "training", "Train model"),
    ("tune", "tuning", "Tune settings"),
    ("register", "registering", "Register student"),
)


def _record_stages(record: dict) -> list[tuple[str, str, str]]:
    """The rows this record has, in run order: every stage its ``stages`` names."""
    stages = record.get("stages") or {}
    return [row for row in _DISTILL_STAGES if row[0] in stages]


def distill_units(record: dict, progress=None) -> list[Unit]:
    """Rows for a distill run: one per stage (label -> train), not one bar split by a label.

    Labelling and training are different things at different granularities — a stage is done
    or it isn't, and the live one is a percentage of *its own* work (frames labelled, or epoch
    N of the total). The run's single bar already folds that into one stage-weighted number
    (``overall_fraction`` / ``fit_progress_fraction``), which is right for "how far along is
    this" and wrong for "which stage is stuck" — that question needs each stage's own unit.

    ``record["stages"]`` is the plan and the history: ``pending`` for a stage not yet reached,
    ``complete`` / ``failed`` / ``cancelled`` for one that is over. The live
    ``DistillProgress`` (``phase`` + ``inner_fraction``) is the only source for how far the
    *running* one has gotten — a page opened after the run started has no live event yet, and
    that stage draws at 0% until the first tick arrives, the same as a processing run's live
    row (``test_a_page_opened_after_the_run_started_is_complete_without_a_live_event``).

    **A live event naming a stage makes it running**, whatever older status the record still
    holds (:func:`_stage_trails`). The coordinator announces a phase *before* it launches the
    stage's job and writes ``running`` only once that job exists — seconds later for labelling,
    which reads every source recording's timestamps first. The page caches the record at the
    phase change, so a cache taken inside that gap holds the stage's previous status for the
    whole stage: ``pending`` on a launch drew "queued" under a moving bar, and ``cancelled`` /
    ``failed`` on a **Resume** drew the resumed training as a failed row.
    """
    phase = getattr(progress, "phase", None)
    stages = record.get("stages") or {}
    units = []
    for key, stage_phase, name in _record_stages(record):
        status = (stages.get(key) or {}).get("status") or "pending"
        if _stage_trails(status, phase, stage_phase):
            status = "running"
        state = STATUS_STATE.get(status)
        if state == DONE:
            units.append(Unit(name, DONE, 1.0, ""))
        elif state == FAILED:
            units.append(
                Unit(name, FAILED, 0.0, _ended(status, {"error": record.get("error")}))
            )
        elif state == RUNNING:
            live = progress if phase == stage_phase else None
            units.append(
                Unit(name, RUNNING, _stage_fraction(live), _stage_status(live))
            )
        else:
            units.append(Unit(name, QUEUED, 0.0, "queued"))
    return units


def _stage_trails(status: str, phase: str | None, stage_phase: str) -> bool:
    """The live event names this stage, and the record has not caught up with it yet.

    Any status but ``running`` is behind a live event — the one it had before this attempt
    (``pending`` on a launch; ``cancelled`` / ``failed`` / ``partial`` on a resume) — except
    ``complete``, which a resume legitimately replays ("reused labeled dataset").
    """
    return phase == stage_phase and status not in ("running", "complete")


def distill_record_lags(record: dict, progress) -> bool:
    """Whether `progress` names a stage `record` has not caught up with — a record read inside
    the coordinator's announce-then-launch gap (see :func:`distill_units`), which a page holding
    it has to re-read: everything it drew off that record is missing the stage's ids."""
    phase = getattr(progress, "phase", None)
    stages = record.get("stages") or {}
    return any(
        _stage_trails(
            (stages.get(key) or {}).get("status") or "pending", phase, stage_phase
        )
        for key, stage_phase, _ in _record_stages(record)
    )


def _stage_fraction(progress) -> float:
    fraction = getattr(progress, "inner_fraction", 0.0) or 0.0
    return min(max(fraction, 0.0), 1.0)


def _stage_status(progress) -> str:
    """The live row's right-hand text: the phase message, this stage's own percentage, and an
    ETA for *this stage* — never the run's, which is stage-weighted and answers "how far is the
    whole distill", not "how much of this stage is left".
    """
    if progress is None:
        return "running"  # record says running; no live event has arrived yet
    from detail.core.progress import format_duration

    message = getattr(progress, "message", "") or "running"
    fraction = _stage_fraction(progress)
    parts = [message]
    if fraction:
        parts.append(f"{fraction * 100:.0f}%")
    eta = getattr(progress, "eta_s", None)
    if eta:
        parts.append(f"~{format_duration(eta)} left")
    return " · ".join(parts)


def export_units(record: dict, progress=None) -> list[Unit]:
    """Rows for an export run: one per **target**, in the order it will do them (#354).

    The sibling of :func:`processing_units`, and it needs to be one rather than a parameter on
    it: that function is recording-shaped down to its field names, and an export's units are
    targets — which is also why `LiveRunPanel.set_units` takes its heading as an argument.

    Same three sources for the same reason. `target_ids` is the plan, including targets the run
    has not reached, which appear in nothing else. `outcomes` is the record, written per target
    as the run goes, and the only place a failure and its reason live — so a page reopened after
    the app restarted is still complete. The live event says which one is in flight: the runner
    counts targets in `recordings_done`, so the target at that index is the running one.

    A finished export writes no `outcomes` for targets it never reached (a cancel stops the
    loop), so "in the plan and not in the outcomes" stays *queued* rather than silently missing.

    **Which target is running is derived from the record, not from a live event.** The runner
    writes `outcomes` as it goes, so the target after the last one recorded is the one in flight
    — which is true on a page reopened tomorrow, where there is no live event at all and the
    rows would otherwise all read *queued* under a bar that is plainly moving. A live `progress`
    only refines it, with the fraction *within* the running target.
    """
    from detail.core.dataclasses import ACTIVE_STATUSES

    outcomes = {o.get("target"): o for o in (record.get("outcomes") or [])}
    target_ids = record.get("target_ids") or list(outcomes)
    active = record.get("status") in ACTIVE_STATUSES
    inner = getattr(progress, "frames_done", 0) or 0
    inner_total = getattr(progress, "frames_total", 0) or 0

    units = []
    for target_id in target_ids:
        outcome = outcomes.get(target_id)
        if outcome is not None and outcome.get("status") == "complete":
            count = len(outcome.get("files") or [])
            units.append(Unit(target_id, DONE, 1.0, f"{count} file(s)"))
        elif outcome is not None:
            units.append(Unit(target_id, FAILED, 0.0, outcome.get("error") or "failed"))
        elif active and len(units) == len(outcomes):
            fraction = inner / inner_total if inner_total else 0.0
            units.append(Unit(target_id, RUNNING, fraction, "writing"))
        else:
            units.append(Unit(target_id, QUEUED, 0.0, "queued"))
    return units


def analysis_units(progress=None, phases: tuple[str, ...] = ()) -> list[Unit]:
    """Rows for an analysis run: one per **phase**, in the module's own declared order (§16).

    The third sibling of :func:`processing_units` / :func:`export_units`, and the one that took
    longest to have a plan to derive from. A processing run's plan is ``recording_ids``, read off
    the record; an analysis run's record has no field like it — a phase does not exist anywhere
    until the run passes through it, which is the whole reason ``AnalysisModule.phases`` (issue
    #393) had to be declared rather than read off a run. ``phases`` **is** the plan, and it comes
    from the module rather than the record — there is no run-record parameter here at all, unlike
    its two siblings, because the record has nothing this derivation needs; the caller resolves
    ``ANALYSIS_MODULES[record["module_id"]].phases`` and passes the result in.

    There is no ``per_recording``-like history either: an analysis run's record does not write
    which phases have already fired, only ``phases`` says what they *are*. So unlike a recording
    row, a done phase carries no duration — that fact is simply not recorded anywhere yet — and
    the whole derivation depends on a **live** event; with none (``progress`` is ``None``, as on
    a page opened with nothing running), every phase reads as queued rather than guessing.

    A phase repeats per recording on a module that treats each recording independently (the
    removed syllable discovery did), so this reads as the *current pass* through the phases, not
    cumulative history across every recording the run has finished — which is also the more
    useful question ("what is recording 3 doing right now") of the two.
    """
    if not phases:
        return []
    message = getattr(progress, "message", None) or ""
    current = message_phase(message, phases)
    units = []
    for i, phase in enumerate(phases):
        if current is not None and i < current:
            units.append(Unit(phase, DONE, 1.0, ""))
        elif current is not None and i == current:
            units.append(Unit(phase, RUNNING, _event_fraction(progress), message))
        else:
            units.append(Unit(phase, QUEUED, 0.0, "queued"))
    return units


def _event_fraction(progress) -> float:
    """The running phase's own bar fraction, from whatever ``frames_done``/``frames_total`` the
    live analysis event carries.

    Not exact for every module: a module that weights several phases into one within-recording
    fraction (the removed syllable discovery's loading/detecting/comparing/clustering split) reports
    this channel for the *recording*, not the phase alone, so a late phase's bar can start already
    part-full. Still monotonic and still moving — which is the bug this exists to not have
    (§6, "frozen at starting") — and exact per-phase accounting is not data any module records.
    """
    total = getattr(progress, "frames_total", 0) or 0
    done = getattr(progress, "frames_done", 0) or 0
    return min(max(done / total, 0.0), 1.0) if total else 0.0


def _ended(status: str, entry: dict) -> str:
    """The text of a row that is over and did not finish: the reason for a failure, and the
    status itself for a cancel or a partial — the word a person needs, since both draw in the
    failure's colour and "cancelled" is not a failure (with the reason after it, if one was
    recorded)."""
    if status == "failed":
        return _failure(entry)
    if entry.get("error"):
        return f"{status} · {_failure(entry)}"
    return status


def _failure(entry: dict) -> str:
    """A failure's reason, short enough for a row.

    The full text is in the record and on the run page; this is the column that has to say
    *which* recording and roughly why at a glance, because in the log it was one line of black
    terminal text among two thousand.
    """
    error = (entry.get("error") or "failed").strip()
    head = error.split("\n", 1)[0]
    # Drop the exception class the record prefixes — "ValueError: no arena defined" reads as
    # noise plus a fact, and the fact is the half a person acts on.
    if ": " in head and head.split(": ", 1)[0].endswith("Error"):
        head = head.split(": ", 1)[1]
    return head if len(head) <= 60 else head[:57] + "…"


def _inner_fraction(inner) -> float:
    done = getattr(inner, "frames_done", 0) or 0
    total = getattr(inner, "frames_total", 0) or 0
    return min(max(done / total, 0.0), 1.0) if total else 0.0


def _running(inner) -> str:
    """The live row's right-hand text: the phase, how far, and an ETA for *this* unit.

    The unit's own ETA, never the run's divided down: on a cohort whose recordings differ by
    hours those are different numbers, and the one a person reads off this row is "how long
    until this recording is done".
    """
    from detail.core.progress import format_duration, remaining_seconds

    message = getattr(inner, "message", "") or "running"
    fraction = _inner_fraction(inner)
    parts = [message]
    if fraction:
        parts.append(f"{fraction * 100:.0f}%")
    eta = getattr(inner, "eta_s", None)
    if eta is None and fraction:
        elapsed = getattr(inner, "elapsed_s", None)
        eta = remaining_seconds(fraction, elapsed) if elapsed else None
    if eta:
        parts.append(f"~{format_duration(eta)} left")
    return " · ".join(parts)
