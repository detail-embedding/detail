"""How a behavior's example clips spread over the cohort — the nudge beside Mark examples (#628).

A behavior marked three times on one animal in one recording has three examples of that animal,
not of the behavior. The search pools examples, so a narrow sample narrows what it finds. These
are **hints, never refusals**: the run config does not change, and one clip still works.

Pure (no Qt, no disk): the dialog hands in what it already holds — the searched recordings'
names and, when the user picked a column to balance across, the cohort metadata table.
"""

from __future__ import annotations

from detail.core.utils import coerce_series_key

#: The example count the setup form already advises (``ADVISED_CLIPS``: 3-5 is measurably
#: steadier). The progress bar fills toward it and turns into a tick there.
TARGET_EXAMPLES = 3


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def subjects(clips) -> tuple[set[str], set[tuple[str, int]]]:
    """The recordings and the individuals (recording, track) ``clips`` come from."""
    recordings = {c["recording_id"] for c in clips}
    individuals = {(c["recording_id"], coerce_series_key(c["track_id"])) for c in clips}
    return recordings, individuals


def spread_text(clips) -> str:
    """A table cell: how many recordings and animals the clips come from, or a dash."""
    if not clips:
        return "—"
    recordings, individuals = subjects(clips)
    return f"{_plural(len(recordings), 'rec')} · {_plural(len(individuals), 'animal')}"


def missing_values(
    clips, recording_ids, metadata: dict[str, dict[str, str]], column: str
) -> list[str]:
    """Values of ``column`` among the searched recordings that no clip comes from, sorted.

    A recording with no value in the column is not a value of its own: an empty cell is a gap
    in the table, and naming it as something to sample from would be a nudge toward nothing.
    """

    def value(recording_id: str) -> str:
        return (metadata.get(recording_id) or {}).get(column, "").strip()

    present = {value(r) for r in recording_ids} - {""}
    covered = {value(c["recording_id"]) for c in clips}
    return sorted(present - covered)


def spread_hint(
    name: str,
    clips,
    recording_names: dict[str, str],
    *,
    metadata: dict[str, dict[str, str]] | None = None,
    column: str | None = None,
) -> str:
    """One sentence naming the first gap in ``name``'s examples, or saying there is none.

    ``recording_names`` is ``{recording_id: name}`` for the recordings being searched — the
    universe a clip could have come from. Gaps in order: all from one recording, all from one
    animal, a value of the balance ``column`` with no clips.
    """
    if not clips:
        return f"{name}: no examples yet."
    recordings, individuals = subjects(clips)
    n = _plural(len(clips), "example")
    if len(recordings) == 1 and len(recording_names) > 1:
        only = next(iter(recordings))
        source = recording_names.get(only, only)
        lead = (
            f"the only example is from {source}"
            if len(clips) == 1
            else f"all {n} are from {source}"
        )
        return f"{name}: {lead}. Add one from another recording."
    if len(individuals) == 1 and len(clips) > 1:
        recording_id, track = next(iter(individuals))
        where = recording_names.get(recording_id, recording_id)
        return (
            f"{name}: all {n} are of one animal (track {track} in {where}). "
            "Add one of another animal if the video has more than one."
        )
    if column and metadata is not None:
        missing = missing_values(clips, recording_names, metadata, column)
        if missing:
            shown = ", ".join(missing[:3]) + (" …" if len(missing) > 3 else "")
            return f"{name}: no examples from {column} = {shown}."
    covered = f", every {column}" if column and metadata is not None else ""
    return (
        f"{name}: {n} from {_plural(len(recordings), 'recording')} and "
        f"{_plural(len(individuals), 'animal')}{covered}."
    )
