"""How an audit's figures read, wherever one is shown (§19).

AUDIT's table, INTERPRET's card and ``detail analysis audit`` each print an estimate, its 95%
confidence interval and how wide that is. They printed them with four hand-written formatters,
and the four disagreed — the table showed a refused estimate's width with a goal check mark, the
CLI said "judged" where the GUI said "audited". This module is the one wording.

**Two rules decide what a figure shows**, both read from the estimate itself:

- **Below** :data:`~detail.storage.behavior_audit.MIN_N_FOR_ESTIMATE` **audited, no estimate.**
  1.00 from three clips reads like 1.00 from fifty. The 95% CI still shows: it is the honest
  statement of how little three clips say.
- **Nothing left to audit, the exact share.** Every clip judged is a count, not a sample, so the
  share is exact, the CI reads "all audited" and the width is 0.00. ``remaining`` is on an
  estimate whose scope is known (``audit_precision(path=)``, ``audit_by_cluster``); without it a
  figure is never called complete.

Pure: no Qt, so the CLI reads it too.
"""

from detail.storage import behavior_audit as audit

#: What the estimate cell says when there is no figure to give.
DASH = "—"


def _audited(estimate: dict | None) -> bool:
    estimate = estimate or {}
    return bool(estimate.get("n")) and bool(estimate.get("interval"))


def complete(estimate: dict | None) -> bool:
    """Whether every clip in the estimate's scope is judged, so its share is exact."""
    estimate = estimate or {}
    return (
        _audited(estimate)
        and not estimate.get("reason")
        and estimate.get("remaining") == 0
    )


def too_few(estimate: dict | None) -> bool:
    """Whether too few clips are audited to give an estimate (the CI still shows)."""
    estimate = estimate or {}
    return (
        _audited(estimate)
        and int(estimate["n"]) < audit.MIN_N_FOR_ESTIMATE
        and not complete(estimate)
    )


def cells(estimate: dict | None) -> tuple[str, str]:
    """The estimate and 95% CI: ``("0.96", "0.80 – 0.99")``, or a dash and a word."""
    estimate = estimate or {}
    if not _audited(estimate):
        return DASH, "not audited"
    if estimate.get("reason"):
        return DASH, "no estimate"
    if complete(estimate):
        return f"{estimate['precision']:.2f}", "all audited"
    low, high = estimate["interval"]
    value = DASH if too_few(estimate) else f"{estimate['precision']:.2f}"
    return value, f"{low:.2f} – {high:.2f}"


def width(estimate: dict | None) -> float | None:
    """How wide the 95% CI is: 0 when nothing is left to audit, ``None`` without a figure."""
    estimate = estimate or {}
    if not _audited(estimate) or estimate.get("reason"):
        return None
    return 0.0 if complete(estimate) else audit.ci_width(estimate)


def goal_reached(estimate: dict | None, goal: float) -> bool:
    """Whether the CI is within ``goal`` wide — a sample-size fact, never a verdict."""
    value = width(estimate)
    return value is not None and value <= float(goal) + 1e-9


def notes(estimate: dict | None) -> list[str]:
    """What qualifies a figure: too few audited, provisional, drawn before the bins were
    recorded, edited after the audit."""
    estimate = estimate or {}
    out = []
    if too_few(estimate):
        out.append(
            f"{estimate['n']} audited: too few for an estimate "
            f"(it takes {audit.MIN_N_FOR_ESTIMATE})"
        )
    if (estimate.get("unjudged_share") or 0) > 0:
        out.append(
            f"provisional: {estimate['unjudged_share']:.0%} of the label not yet audited"
        )
    unbinned = int(estimate.get("n_drawn_before_bins") or 0)
    if unbinned:
        out.append(
            f"{unbinned} audited clip{'s were' if unbinned != 1 else ' was'} drawn before "
            "this behavior's score bins were recorded, and "
            f"{'are' if unbinned != 1 else 'is'} weighted by them"
        )
    before = int(estimate.get("n_before") or 0)
    if estimate.get("since") and before:
        out.append(
            f"edited after the audit: {int(estimate.get('n') or 0)} audited since, "
            f"{before} before"
        )
    return out


def line(estimate: dict | None) -> str:
    """One estimate as a line of text, refusals included: what the CLI prints."""
    estimate = estimate or {}
    if not estimate.get("n"):
        return "not audited"
    if estimate.get("reason"):
        pct = round(100 * (estimate.get("coverage") or 0))
        return f"no estimate ({pct}% of the audit still applies)"
    if complete(estimate):
        return f"precision {estimate['precision']:.2f}  (all {estimate['n']} audited)"
    value, _interval = cells(estimate)
    low, high = estimate["interval"]
    return f"precision {value}  (95% CI {low:.2f}-{high:.2f}, n={estimate['n']})"


def query_review_line(name: str, cutoff: float, estimate: dict | None, rows) -> str:
    """One behavior's review on a query run (§20): its figure once every recording is exact at
    ``cutoff``, else how many are — and the exact figure over the range that was reviewed, never
    a figure at the cutoff (the answered clips are the top of the candidates, not a sample).
    """
    rows = list(rows)
    if estimate is not None and estimate.get("exact"):
        return (
            f"{name}: {line(estimate)}, every recording reviewed at {cutoff:.2f} "
            f"({len(rows)})"
        )
    done = sum(1 for row in rows if row["exact"])
    reached = ""
    if estimate is not None and estimate.get("at_floor") is not None:
        reached = (
            f"; reviewed to {estimate['reviewed_floor']:.2f}: "
            f"{line(estimate['at_floor'])}"
        )
    return f"{name}: {done} of {len(rows)} recordings reviewed at {cutoff:.2f}{reached}"


def query_review_cell(row: dict) -> str:
    """One recording's review of a behavior (``QueryRun.review_by_recording``'s row)."""
    own = (
        f" · {row['examples']} on your own examples, not reviewed"
        if row["examples"]
        else ""
    )
    if not row["size"]:
        return f"no clips at this cutoff{own}"
    if row["exact"]:
        return f"{line(row['estimate'])} · {row['removed']} removed{own}"
    if row["at_floor"] is not None:
        return (
            f"reviewed to {row['reviewed_floor']:.2f} · "
            f"{row['estimate']['remaining']} not reviewed below it{own}"
        )
    return f"not reviewed{own}"
