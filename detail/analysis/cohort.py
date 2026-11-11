"""What a cohort's metadata says about itself, before anything is measured.

A comparison asks the user which column to compare across and what one observation is (§24).
Both questions are unanswerable from a bare list of column names — which is what the first
Comparisons screen offered, and a user reading it could not tell what the options were or how
to choose between them.

Everything here is derived from ``metadata.csv`` alone, so it is available *before* a run rather
than after. That is the whole point: ``subject`` and ``window_id`` come off the measures table,
which does not exist until the comparison has been computed, so any help offered at the moment of
choosing has to come from the cohort table.

Pure and Qt-free, so the reasoning is testable without a ``QApplication`` and reachable from the
CLI, not only the dialog that prompted it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

#: Column names that suggest an animal identity rather than a manipulation. A *tiebreak only* —
#: the structural signals below decide, and a cohort that names its animals `panda` still works.
_SUBJECT_NAME_HINTS = (
    "animal",
    "animal_id",
    "subject",
    "subject_id",
    "mouse",
    "mouse_id",
    "rat",
    "rat_id",
    "cage",
    "cohort",
)

#: Above this many distinct values, listing them is noise, so the summary reports the shape.
_MAX_LISTED_VALUES = 4


@dataclass(frozen=True)
class ColumnSummary:
    """One metadata column, described by what is actually in it.

    ``counts`` excludes blanks deliberately: a blank is stored as ``""`` and
    :func:`detail.analysis.compare._factors` turns a missing value into that same ``""``, so
    counting blanks as a level would report a group that is really "the ones nobody filled in".
    ``blank`` carries them separately so a surface can say so.
    """

    name: str
    counts: Mapping[str, int]
    blank: int

    @property
    def n_values(self) -> int:
        """How many distinct non-blank values."""
        return len(self.counts)

    @property
    def n_recordings(self) -> int:
        return sum(self.counts.values()) + self.blank

    @property
    def has_repeats(self) -> bool:
        """Whether some value covers more than one recording — the shape that makes averaging
        within this column a different answer from counting each recording."""
        return any(count > 1 for count in self.counts.values())

    @property
    def usable_for_grouping(self) -> str | None:
        """``None`` when this column can define groups, else why it cannot.

        Only a column with fewer than two values is refused. A column with *many* values is
        not — comparing two recordings one-against-one is a small study, not an invalid one,
        and it is exactly the shape of a pilot pair. The surface says how small; it does not
        decide for the user.
        """
        if self.n_values == 0:
            return "no values recorded"
        if self.n_values == 1:
            only = next(iter(self.counts))
            return f"every recording says {only!r}"
        return None

    def describe(self) -> str:
        """The values, as a surface should show them beside the column's name."""
        if not self.counts:
            return "no values recorded"
        ranked = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if self.n_values > _MAX_LISTED_VALUES:
            sizes = set(self.counts.values())
            shape = (
                f"{next(iter(sizes))} each"
                if len(sizes) == 1
                else f"{min(sizes)}–{max(sizes)} recordings each"
            )
            return f"{self.n_values} values, {shape}"
        listed = ", ".join(f"{value} ({count})" for value, count in ranked)
        return f"{listed} + {self.blank} blank" if self.blank else listed


@dataclass(frozen=True)
class GroupPreview:
    """One group a comparison would produce, sized before it is run."""

    group: tuple[str, ...]
    n_recordings: int
    n_subjects: int | None

    @property
    def label(self) -> str:
        return " | ".join(self.group) if self.group else "(all)"


def _values(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    column: str,
) -> list[tuple[str, str]]:
    """``(recording_id, value)`` for each recording, blanks included as ``""``."""
    return [(r, (metadata.get(r, {}).get(column) or "").strip()) for r in recordings]


def describe_columns(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    columns: Iterable[str],
) -> list[ColumnSummary]:
    """Each column summarized over ``recordings``, in the order given."""
    out = []
    for column in columns:
        pairs = _values(metadata, recordings, column)
        counts = Counter(value for _r, value in pairs if value)
        blank = sum(1 for _r, value in pairs if not value)
        out.append(ColumnSummary(name=column, counts=dict(counts), blank=blank))
    return out


def _levels(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    column: str,
) -> dict[str, set[str]]:
    """``{value: {recording_id, ...}}`` for one column, blanks dropped."""
    out: dict[str, set[str]] = {}
    for recording, value in _values(metadata, recordings, column):
        if value:
            out.setdefault(value, set()).add(recording)
    return out


def _determines(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    source: str,
    target: str,
) -> bool:
    """Whether each value of ``source`` maps to exactly one value of ``target``.

    A subject column determines its own stable covariates — one animal has one sex — while a
    treatment column determines nothing, because the same drug is given to different animals.
    That asymmetry is the sharpest metadata-only signal for which of two columns is an identity.
    """
    seen: dict[str, str] = {}
    for recording, value in _values(metadata, recordings, source):
        if not value:
            continue
        other = (metadata.get(recording, {}).get(target) or "").strip()
        if not other:
            continue
        if seen.setdefault(value, other) != other:
            return False
    return bool(seen)


def _crossing(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    column: str,
    group_by: Sequence[str],
) -> int:
    """How many of ``column``'s values appear under more than one compared group.

    An animal measured on both vehicle and drug is a crossover design, and it is the
    strongest evidence available that the column is an identity rather than a manipulation: a
    *treatment* almost never partitions its own recordings across another treatment's levels
    while also repeating.
    """
    if not group_by:
        return 0
    crossed = 0
    for members in _levels(metadata, recordings, column).values():
        groups = {
            tuple((metadata.get(r, {}).get(g) or "").strip() for g in group_by)
            for r in members
        }
        if len(groups) > 1:
            crossed += 1
    return crossed


def infer_subject_column(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    *,
    group_by: Sequence[str] = (),
    columns: Iterable[str] | None = None,
) -> str | None:
    """The column that identifies a repeated subject across recordings, or ``None``.

    ``None`` means every recording stands alone — nobody contributed two — and the unit-of-
    analysis question does not arise, so a surface should not ask it.

    A candidate must be **complete**: a column with any blank is refused outright, because
    ``compare._factors`` turns a missing value into ``""`` and every unfilled recording would
    silently pool into one convincing-looking extra animal.

    This is a *suggestion*, and every surface using it must show which column it picked. It is
    inferred rather than configured because nothing in ``metadata.csv`` marks a column's role
    (`storage/metadata.py` reserves exactly one name, ``recording_id``), and asking every user to
    declare one up front is a setup step paid by everybody to help the few cohorts that repeat.
    """
    if columns is None:
        columns = sorted({key for row in metadata.values() for key in row})
    group_by = list(group_by)

    scored = []
    for summary in describe_columns(metadata, recordings, columns):
        column = summary.name
        if column in group_by or column == "recording_id":
            continue
        if summary.blank or not summary.has_repeats:
            continue
        determines = sum(
            1
            for other in columns
            if other != column and _determines(metadata, recordings, column, other)
        )
        determined_by = sum(
            1
            for other in columns
            if other != column and _determines(metadata, recordings, other, column)
        )
        # A *constant* column is structurally indistinguishable from a single-animal
        # crossover — `subject`, which is the track index `0` in every experiment that
        # exists, repeats and crosses every group exactly the way one animal measured under
        # two treatments does. What separates them is that a real identity **explains
        # something else**: an animal has a sex and an id, while a track slot explains
        # nothing. So a one-value column earns its place only by determining another.
        if summary.n_values < 2 and determines == 0:
            continue
        scored.append(
            (
                _crossing(metadata, recordings, column, group_by),
                determines - determined_by,
                column.lower() in _SUBJECT_NAME_HINTS,
                -summary.n_values,  # fewer subjects, each with more recordings, first
                column,
            )
        )
    if not scored:
        return None
    # `max` over the tuple: crossing, then the determination asymmetry, then the name hint.
    # The trailing name keeps the choice stable when a cohort carries two equivalent identity
    # columns (`animal` beside `animal_id`), which is common and must not flip between runs.
    return max(scored)[-1]


def deferred_columns(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    group_by: Sequence[str],
) -> list[str]:
    """Grouping columns ``metadata.csv`` does not carry, so their groups are not knowable yet.

    ``subject`` and ``window_id`` are the ones a comparison offers on purpose: they come from the
    behavior artifact's own rows and from the window spec, and are resolved when the comparison
    runs. A mistyped column lands here too, which is the right answer for it as well.
    """
    seen: set[str] = set()
    for recording in recordings:
        seen |= set(metadata.get(recording, {}))
    return [column for column in group_by if column not in seen]


def group_preview(
    metadata: Mapping[str, Mapping[str, str]],
    recordings: Sequence[str],
    *,
    group_by: Sequence[str],
    subject_column: str | None = None,
) -> list[GroupPreview]:
    """The groups a comparison would produce, sized — before measuring anything.

    ``n_subjects`` is ``None`` when no subject column is known, rather than equal to
    ``n_recordings``: "one animal per recording" and "we do not know" are different facts and a
    surface should not render the second as the first.

    Returns ``[]`` when any grouping column is one :func:`deferred_columns` names — the same
    principle one level up. Reading a column no metadata row has gave every recording the empty
    string, so grouping by ``subject`` previewed as *one* blank group holding the whole cohort:
    a confident, wrong preview of a comparison that will actually produce one group per subject.
    """
    if deferred_columns(metadata, recordings, group_by):
        return []
    buckets: dict[tuple[str, ...], list[str]] = {}
    for recording in recordings:
        row = metadata.get(recording, {})
        key = tuple((row.get(column) or "").strip() for column in group_by)
        buckets.setdefault(key, []).append(recording)

    out = []
    for key in sorted(buckets):
        members = buckets[key]
        subjects = (
            len(
                {
                    (metadata.get(r, {}).get(subject_column) or "").strip()
                    for r in members
                }
            )
            if subject_column
            else None
        )
        out.append(
            GroupPreview(group=key, n_recordings=len(members), n_subjects=subjects)
        )
    return out
