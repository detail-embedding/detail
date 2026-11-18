"""``clip_review.csv`` — a clustering run's precision, from reviewing every clip of a cluster (§19/§22).

The successor of the one-by-one ``audit.csv`` review on a temporal clustering run. A reviewer opens
one cluster, watches **every** clip of it, takes out the ones that are not its label
(``storage/clip_edits.py``) and finishes. Finishing writes a **census**: one answer per clip of the
cluster, ``no`` for a clip taken out of the label or moved to another, ``yes`` for every other one,
all in one append (:func:`append_answers`). So a reviewed cluster's precision is a count, not a
sample.

Columns ``recording_id, subject, t_ns, end_t_ns, node, label_path, drawn_at, answer``. A clip is a
leaf run (``ClipRef``); ``node`` is the labelled cluster it counted toward and ``label_path`` that
cluster's label path when it was judged. ``answer`` is ``yes`` / ``no``, or ``withdrawn``: the
clip was put back in its label after the answer, so the answer no longer describes it and the clip
reads as not yet reviewed (:func:`withdraw`). ``drawn_at`` is when the
row was written (ISO-8601 UTC), which is what "judged since the label's clusters changed" compares
(``membership_changed_at``); the name is kept from the random draw this file first recorded.

**Rows from the earlier draw stay readable.** Before the census, a reviewer judged 25 clips drawn
at random, and a draw logged its clips with an empty ``answer`` first. Such a row is read and
ignored (it judges nothing); a judged row from a draw still counts as that clip's answer until a
census supersedes it — but a cluster reads as reviewed only once every one of its clips has one.

Appended through ``storage/append_log.AppendLog``, never rewritten, so a crash costs at most the
tail of the batch being written. Like ``audit.csv`` it carries no ``schema_version`` row: its
header is its schema, and ``AppendLog`` reads a row against the columns it was written under.
``audit.csv`` is left exactly as it was and is still read by the earlier audit's readers; nothing
here reads it.
"""

from dataclasses import dataclass
from pathlib import Path

from detail.core.utils import utc_timestamp
from detail.storage.append_log import AppendLog
from detail.storage.clip_edits import ClipRef
from detail.storage.clip_edits import clip_ref

REVIEW_FILENAME = "clip_review.csv"
REVIEW_COLUMNS = [
    "recording_id",
    "subject",
    "t_ns",
    "end_t_ns",
    "node",
    "label_path",
    "drawn_at",
    "answer",
]
YES = "yes"
NO = "no"
#: Written for a clip put back in its label (``put_back_clips``): supersedes the clip's answer
#: under that path without being one. The owner's rule (#24): the census said "no" because the
#: clip was taken out; once it is back, that answer describes a clip that is not there.
WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class ReviewRow:
    """One row of ``clip_review.csv``: an answer, or (``answer == ""``) an earlier draw's log."""

    recording_id: str
    subject: int
    t_ns: int
    end_t_ns: int
    node: int
    label_path: str
    drawn_at: str
    answer: str

    @property
    def clip(self) -> ClipRef:
        return ClipRef(self.recording_id, self.subject, self.t_ns, self.end_t_ns)

    @property
    def judged(self) -> bool:
        return self.answer in (YES, NO)

    @property
    def withdrawn(self) -> bool:
        return self.answer == WITHDRAWN

    @property
    def yes(self) -> bool | None:
        """``True`` / ``False`` for an answer, ``None`` for a draw."""
        return None if not self.judged else self.answer == YES


def review_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / REVIEW_FILENAME


def _log(run_dir: Path | str) -> AppendLog:
    return AppendLog(review_path(run_dir), REVIEW_COLUMNS)


def append_answers(
    run_dir: Path | str,
    answers,
    *,
    node: int,
    label_path: str,
    drawn_at: str | None = None,
) -> list[ReviewRow]:
    """Append one answer per ``(clip, yes)`` in ``answers``, in one durable write.

    Every row carries the same ``node``, ``label_path`` and time: a census is one decision about
    one cluster. Returns the rows written.
    """
    stamp = drawn_at or utc_timestamp()
    rows = []
    for clip, yes in answers:
        ref = clip_ref(clip)
        rows.append(
            ReviewRow(
                recording_id=ref.recording_id,
                subject=ref.subject,
                t_ns=ref.t_ns,
                end_t_ns=ref.end_t_ns,
                node=int(node),
                label_path=str(label_path),
                drawn_at=stamp,
                answer=YES if yes else NO,
            )
        )
    return append_rows(run_dir, rows)


def append_rows(run_dir: Path | str, rows: list[ReviewRow]) -> list[ReviewRow]:
    """Append ``rows`` as they are, in one durable write — a carried answer keeps its own
    ``drawn_at``. Returns the rows written."""
    if not rows:
        return []
    _log(run_dir).append_many(
        [
            [
                row.recording_id,
                row.subject,
                row.t_ns,
                row.end_t_ns,
                row.node,
                row.label_path,
                row.drawn_at,
                row.answer,
            ]
            for row in rows
        ]
    )
    return rows


def read_review(run_dir: Path | str) -> list[ReviewRow]:
    """Every row, in the order written — answers, superseded ones kept, and any draw rows an
    earlier version logged."""
    out = []
    for row in _log(run_dir).rows():
        try:
            out.append(
                ReviewRow(
                    recording_id=row.get("recording_id") or "",
                    subject=int(row.get("subject") or 0),
                    t_ns=int(row["t_ns"]),
                    end_t_ns=int(row["end_t_ns"]),
                    node=int(row.get("node") or -1),
                    label_path=row.get("label_path") or "",
                    drawn_at=row.get("drawn_at") or "",
                    answer=(row.get("answer") or "").strip().lower(),
                )
            )
        except (KeyError, ValueError):
            continue  # a torn final row: AppendLog's guarantee is that only the last can be
    return out


def latest_by_path(rows: list[ReviewRow]) -> dict[tuple, tuple[int, ReviewRow]]:
    """``{(clip key, label path): (position, its latest answer there)}``.

    An answer is a judgment of a clip **as one label**: "is this climb?". So it is kept per
    label path, and a clip judged as climb and later as rear has one answer for each — the
    rear one does not overwrite what was said about climb, and the climb one is not read as an
    answer about rear. ``position`` is the row's place in the log, so :func:`answer_for` can
    tell which of two candidates is the later. A withdrawal is kept here as the latest row too,
    so it supersedes the answer before it (:func:`answer_for` reads it as no answer).
    """
    out: dict[tuple, tuple[int, ReviewRow]] = {}
    for position, row in enumerate(rows):
        if row.judged or row.withdrawn:
            out[(row.clip.key, row.label_path)] = (position, row)
    return out


def answer_for(latest: dict, key: tuple, label_path: str) -> ReviewRow | None:
    """The answer for clip ``key`` judged as ``label_path``, from :func:`latest_by_path`.

    A row with no ``label_path`` (none should exist: the column was there from the start, but a
    hand-edited or torn file can leave it blank) says nothing about which label it judged, and
    is read as it always was — an answer for the clip under any label — unless a later answer
    names this path.
    """
    candidates = [
        found
        for found in (latest.get((key, label_path)), latest.get((key, "")))
        if found is not None
    ]
    if not candidates:
        return None
    row = max(candidates, key=lambda pair: pair[0])[1]
    return None if row.withdrawn else row


def withdraw(
    run_dir: Path | str, keys, *, drawn_at: str | None = None
) -> list[ReviewRow]:
    """Withdraw every current answer about the clips ``keys`` (``ClipRef.key``s), under every
    label path it was given as, in one durable write. Returns the rows written.

    What putting a clip back does to its review: the clip reads as not yet reviewed again, so the
    cluster it is in is not "all reviewed" on an answer about a clip that is no longer out.
    """
    wanted = set(keys)
    stamp = drawn_at or utc_timestamp()
    rows = [
        ReviewRow(
            recording_id=row.recording_id,
            subject=row.subject,
            t_ns=row.t_ns,
            end_t_ns=row.end_t_ns,
            node=row.node,
            label_path=row.label_path,
            drawn_at=stamp,
            answer=WITHDRAWN,
        )
        for (key, _path), (_position, row) in latest_by_path(
            read_review(run_dir)
        ).items()
        if key in wanted and row.judged
    ]
    return append_rows(run_dir, rows)
