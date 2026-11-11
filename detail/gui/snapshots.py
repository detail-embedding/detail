"""Pinned artifacts on screen (§21) — what left a run, and what it was.

A snapshot is a frozen copy of an authored run's §19 artifact, taken when a number leaves: an
export, a comparison, a finalize. Until this they were invisible outside the CLI — pressing
**Finish** wrote one and nothing on screen said so, which makes a guarantee nobody can see.

**One surface, deliberately.** The first attempt paired a short table with a dialog, and the dialog
had nothing to add: the row already carried the settings, so opening one showed a code hash and the
same instance list BROWSE already shows. A row that opens a detail view has to *earn* the cue
(``docs/gui_design.md`` §3), so the table carries the whole story instead — including the
vocabulary, which is what a person is actually trying to recall.

Double-clicking a row opens the interpretation itself, read-only, in the page the row is on
(§21). The columns between "Why" and "Instances" are the **module's**, not this file's: clustering
answers with a granularity and its names, Query with its behaviors and their cutoffs. Hardcoding
clustering's shape here is what made a fully calibrated Query run read ``— · 0 · unnamed``.
"""

BASE_COLUMNS = ("Taken", "Why")
TAIL_COLUMNS = ("Instances", "Used by")


def snapshot_columns(run, records: list[dict]) -> list[str]:
    """The table's headers: the columns every snapshot has, plus the module's own.

    Asked of the module rather than fixed here. The first cut hardcoded clustering's
    (Clusters · Named · Vocabulary), which meant a calibrated Query run — two named behaviors, both
    with cutoffs — rendered ``— · 0 · unnamed``. Not a blank row but a *wrong* one, and nothing
    failed, because an empty cell is a plausible cell.
    """
    middle = ()
    if records:
        middle = run.snapshot_interpretation(records[0]).columns
    return [*BASE_COLUMNS, *middle, *TAIL_COLUMNS]


def snapshot_rows(experiment, run) -> list[tuple]:
    """One row per pinned artifact, oldest first — the whole story of what left the run.

    Rich enough to answer "which of these did the dose comparison use, and what was it?" without
    opening anything, because that is the question these exist for.

    ``Used by`` is **derived** from the comparisons rather than stored on the snapshot (§21): a
    fact checkable from the store gets no second writer, and neither an export path nor a
    not-yet-allocated comparison id could have been recorded at pin time anyway.
    """
    used = experiment.comparisons_using(run.id)
    rows = []
    for record in run.snapshots():
        snapshot_id = str(record.get("snapshot_id"))
        try:
            instances = len(run.snapshot_artifact(snapshot_id).instances)
        except (FileNotFoundError, ValueError):
            instances = None
        consumers = used.get(snapshot_id, [])
        rows.append(
            (
                str(record.get("taken_at", ""))[:16].replace("T", " "),
                str(record.get("reason", "")),
                *run.snapshot_interpretation(record).cells,
                "—" if instances is None else f"{instances:,}",
                ", ".join(consumers) if consumers else "—",
            )
        )
    return rows
