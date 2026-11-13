"""Pure per-epoch training-history computations, shared by the GUI curve and the CLI.

Kept Qt-free (no ``pyqtgraph``/``PyQt6`` import) so `detail/cli.py` can report the same
"best so far" fact `gui/training_curve.py` draws, without pulling Qt into the CLI process.
"""

from __future__ import annotations


def best_epoch(history: list[dict]) -> tuple[int, float] | None:
    """The highest box mAP50 in the history and the epoch it was reached, or None.

    ``max`` on the metric rather than the last row: the last epoch is not the best one on any run
    that has started to overfit, and `best.pt` is saved from the peak — so reporting the tail
    would name a score the saved weights do not have.

    Ties go to the **earliest** epoch, because that is the one that answers "how long ago did
    this stop improving". `max` keeps the first maximum it sees, so oldest-first input gives that
    for free; the ordering is the contract, hence the note.
    """
    scored = [
        (row["epoch"], row["box_map50"])
        for row in history
        if row.get("box_map50") is not None and row.get("epoch") is not None
    ]
    if not scored:
        return None
    best = max(scored, key=lambda pair: pair[1])
    return int(best[0]), float(best[1])


def best_line(history: list[dict], *, total_epochs: int | None = None) -> str:
    """The sentence under the curve.

    Names the epoch as well as the score, because "0.847" alone cannot be compared with anything
    — what makes it actionable is that it was reached at epoch 31 and the run is on epoch 58.
    """
    best = best_epoch(history)
    if best is None:
        return "(no completed epoch yet)"
    epoch, score = best
    latest = max(
        (row["epoch"] for row in history if row.get("epoch") is not None), default=0
    )
    text = f"Best so far: {score:.3f} mAP50 at epoch {epoch}"
    if latest > epoch:
        # The gap is the whole reason the epoch is named. Stated as a count of epochs rather
        # than as "plateaued", which would be the tool making the call.
        behind = latest - epoch
        text += f" — no improvement in {behind} epoch{'s' if behind != 1 else ''}"
    if total_epochs:
        text += f" (of {total_epochs})"
    return text
