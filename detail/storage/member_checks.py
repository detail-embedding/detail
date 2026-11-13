"""Clip checks on a clustering run (§22 *Clip checks*): one person's answer, per clip, to "does
this clip fit the cluster?", kept beside the run before any label exists.

Append-only through ``AppendLog`` (human judgment is primary data, Tenet 9): a changed answer is
a new row, and a clip's **latest** row is its answer. The figure a cluster gets is
``fits / (fits + doesnt)`` with the 95% Wilson interval; ``unclear`` is counted, in neither.
Nothing reads these to build anything: they score the clustering, never feed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from detail.core.utils import subject_key, utc_timestamp
from detail.storage.append_log import AppendLog

FILENAME = "member_checks.csv"
COLUMNS = [
    "node",
    "recording_id",
    "subject",
    "start_ns",
    "end_ns",
    "verdict",
    "checked_at",
]
FITS, DOESNT, UNCLEAR = "fits", "doesnt", "unclear"
VERDICTS = (FITS, DOESNT, UNCLEAR)


@dataclass(frozen=True)
class ClipKey:
    """One clip of one cluster, as the clip window names it."""

    node: int
    recording_id: str
    subject: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class CheckSummary:
    """A cluster's checked clips: the counts, the latest answer per clip, and the estimate
    (``None`` until a clip fits or does not)."""

    fits: int
    doesnt: int
    unclear: int
    estimate: float | None
    interval: tuple[float, float] | None
    answers: dict[ClipKey, str]

    @property
    def checked(self) -> int:
        return self.fits + self.doesnt + self.unclear


def _log(run_dir: Path | str) -> AppendLog:
    return AppendLog(Path(run_dir) / FILENAME, COLUMNS)


def append_check(run_dir: Path | str, key: ClipKey, verdict: str) -> None:
    """Record one answer, durably.

    Raises:
        ValueError: ``verdict`` is not one of :data:`VERDICTS`.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, not {verdict!r}")
    _log(run_dir).append(
        [
            int(key.node),
            str(key.recording_id),
            subject_key(key.subject),
            int(key.start_ns),
            int(key.end_ns),
            verdict,
            utc_timestamp(),
        ]
    )


def summary(run_dir: Path | str, node: int) -> CheckSummary:
    """The node's checks, latest answer per clip. A run with no checks has an empty summary."""
    log = _log(run_dir)
    answers: dict[ClipKey, str] = {}
    if log.path.exists():
        for row in log.rows():
            try:
                key = ClipKey(
                    int(row["node"]),
                    str(row["recording_id"]),
                    subject_key(row["subject"]),
                    int(row["start_ns"]),
                    int(row["end_ns"]),
                )
            except (KeyError, TypeError, ValueError):
                continue  # a torn last row from a crash: the append log tolerates it
            if key.node == int(node) and row.get("verdict") in VERDICTS:
                answers[key] = row["verdict"]
    return summarize(answers)


def summarize(answers: dict[ClipKey, str]) -> CheckSummary:
    """The counts and estimate over latest answers — what :func:`summary` reads from disk,
    and what a window recomputes after an answer without reading the file again."""
    from detail.storage.behavior_audit import wilson_interval

    counts = {v: sum(1 for a in answers.values() if a == v) for v in VERDICTS}
    judged = counts[FITS] + counts[DOESNT]
    return CheckSummary(
        fits=counts[FITS],
        doesnt=counts[DOESNT],
        unclear=counts[UNCLEAR],
        estimate=counts[FITS] / judged if judged else None,
        interval=wilson_interval(counts[FITS], judged),
        answers=answers,
    )
