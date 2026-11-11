"""Which upstream run an analysis read in each recording (§6.5 "Upstream run specs").

An analysis over a selection reads **each recording's own** upstream run. Those runs share
settings, not an id: a per-recording id is a counter (``emb_01``, ``emb_02``) that a retry, a
deleted run or a separate launch moves, so the same settings routinely sit under different ids
in different recordings. A run therefore records its selection as a per-recording map,
``<ref>s`` beside the single ``<ref>``, which is set only when every recording's id agrees.

A run recorded before the map existed carries the single id alone, and it meant that id in
every recording. :func:`run_for_recording` answers both shapes, so a reader never has to know
which one it holds.
"""

from collections.abc import Mapping


def uniform_run_id(run_ids: Mapping[str, str]) -> str | None:
    """The one id every recording's run shares, or ``None`` when they differ (or none)."""
    distinct = set(run_ids.values())
    return next(iter(distinct)) if len(distinct) == 1 else None


def run_for_recording(
    source, recording_id: str, ref_field: str = "embed_run_id"
) -> str | None:
    """``recording_id``'s own upstream run in ``source`` — a job, a state, a record's
    ``params`` or a ``data_sources()`` dict alike.

    The per-recording map (``<ref_field>s``) wins; a source without one is a run recorded
    before it existed, whose single ``ref_field`` meant that id in every recording. ``None``
    when the source names no run for this recording.
    """
    if isinstance(source, Mapping):
        run_ids, run_id = source.get(f"{ref_field}s"), source.get(ref_field)
    else:
        run_ids = getattr(source, f"{ref_field}s", None)
        run_id = getattr(source, ref_field, None)
    if run_ids:
        found = run_ids.get(recording_id)
        return str(found) if found else None
    return str(run_id) if run_id else None


def require_run_for_recording(
    source, recording_id: str, ref_field: str = "embed_run_id"
) -> str:
    """:func:`run_for_recording`, for a reader that cannot go on without one.

    Raises:
        KeyError: ``source`` names no run for ``recording_id`` — a recording outside the
            run's selection.
    """
    run_id = run_for_recording(source, recording_id, ref_field)
    if run_id is None:
        raise KeyError(
            f"this run read no {ref_field.removesuffix('_id')} in recording "
            f"{recording_id!r}; it is not one of the run's recordings"
        )
    return run_id


def runs_by_recording(
    source, recording_ids, ref_field: str = "embed_run_id"
) -> dict[str, str]:
    """``{recording_id: run_id}`` over ``recording_ids``, omitting any the source names no run
    for — :func:`run_for_recording` applied to a selection."""
    out: dict[str, str] = {}
    for recording_id in recording_ids:
        run_id = run_for_recording(source, recording_id, ref_field)
        if run_id is not None:
            out[recording_id] = run_id
    return out
