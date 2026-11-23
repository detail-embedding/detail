"""Seeding an earlier one-by-one audit (``audit.csv``) in a test, the way it was drawn and judged.

The review of every clip replaced the one-by-one draw and judge on both run kinds (§19, §20, §22):
``AuthoredRun.audit_sample`` and ``label_audit`` are gone, and nothing in the app writes
``audit.csv`` any more. Every reader of it stays — a run audited before keeps what it bought,
shown as *Earlier one-by-one audit* — so its tests still need one to read. :func:`draw` and
:func:`judge` write it the way the removed methods did, through the real storage writer
(``behavior_audit.append_audit``), minus their refusals: a test seeding a run's history is not
an edit of it.

``draw_sample``, ``_strata`` and ``fix_bins`` are the removed draw, moved here verbatim, so an
audit a test seeds is drawn exactly as a real one was — a query run's bins recorded in
``audit.yaml`` at its first draw — and every estimate over it reads as before.
"""

from pathlib import Path

import numpy as np

from detail.storage import behavior_audit as audit
from detail.storage import behavior_output as bo
from detail.storage import state_yaml
from detail.storage.behavior_audit import _label_edges, _score_stratum

DEFAULT_BUDGET = 25


def fix_bins(
    run_dir: Path | str, edges: dict[str, np.ndarray], *, seed: int
) -> dict[str, np.ndarray]:
    """Record each label's score-bin edges in ``audit.yaml`` unless it already has some.

    Returns the edges in force for every label passed — the recorded ones where a label had
    them: once a label has been drawn from, its bins never move. Written atomically,
    ``schema_version`` first, through the authored-state writer. The stamp is read through
    ``behavior_audit.utc_timestamp``, where it lived, so a test that pins that clock pins this.
    """
    path = Path(run_dir) / bo.AUDIT_BINS_FILENAME
    record = state_yaml.read_state_yaml(path) if path.is_file() else {}
    stored = dict(record.get("bins") or {})
    missing = [label for label in edges if str(label) not in stored]
    if missing:
        now = audit.utc_timestamp()
        for label in missing:
            stored[str(label)] = {
                "edges": [float(e) for e in edges[label]],
                "seed": int(seed),
                "fixed_at": now,
            }
        state_yaml.write_state_yaml(
            path, {"stratified_by": "score_bin", "bins": stored}
        )
    return {
        str(label): np.asarray(stored[str(label)]["edges"], dtype=np.float64)
        for label in edges
    }


def _strata(
    instances, stratum_of: dict | None = None, edges: dict | None = None
) -> tuple[str, dict]:
    """Group the bouts into the strata a sample should spread across.

    Two shapes, decided by the artifact rather than by the caller:

    - **Scored** (a query run) → **score bins, per label**: ``(source_label, bin)``, cut at
      ``edges[source_label]`` (:func:`fix_bins`) or, for a label without recorded edges, at
      equal widths over that label's own scores. Precision becomes a function of confidence,
      which is what lets one audit serve every cutoff. Per label because a draw across several
      behaviors at once must cut each behavior's scores where a draw of that behavior alone
      would; pooled edges put one behavior's whole range into a bin or two of another's.
    - **Unscored** (a clustering run) → **one stratum per label**. A cluster carries no
      confidence, so there is nothing to bin; its natural population is itself, and the answer
      is precision per named cluster.

    Without the second shape a clustering audit draws **nothing at all** — the scored path
    skips scoreless bouts by design, so `draw_sample` returned an empty list for every
    clustering artifact while looking like it had merely found nothing to sample.

    Strata are keyed by label path where there is one, so two nodes given one name are one
    stratum, as they are one row everywhere else. Which bouts may be drawn at all is the
    caller's: a clustering run's unlabelled remainder makes no claim to confirm, and
    ``AuthoredRun.audit_sample`` leaves it out of the pool before it gets here.
    """
    bouts = list(bo.bouts(instances))
    scored = [row for row in bouts if row.score is not None]
    if scored:
        label_edges = _label_edges(scored, edges)
        groups: dict = {}
        for row in scored:
            groups.setdefault(_score_stratum(row, label_edges), []).append(row)
        return "score_bin", groups
    groups = {}
    if stratum_of is not None:
        # A clustering run with a node cache: one stratum per **cluster**, so each one a label
        # spans gets clips of its own and a figure of its own. A bout the cache cannot place falls
        # back to its label's stratum rather than out of the draw.
        for row in bouts:
            key = stratum_of.get(row.instance_id)
            groups.setdefault(
                key if key is not None else (row.label_path or row.source_label), []
            ).append(row)
        return "cluster", groups
    for row in bouts:
        groups.setdefault(row.label_path or row.source_label, []).append(row)
    return "label", groups


def draw_sample(
    instances,
    *,
    budget: int = DEFAULT_BUDGET,
    seed: int = 0,
    exclude_instance_ids=(),
    stratum_of: dict | None = None,
    edges: dict | None = None,
) -> list:
    """``budget`` instances spread across the artifact's strata, deterministically.

    ``edges`` (``{source_label: edges}``, :func:`fix_bins`) are where a scored artifact's bins
    are cut; the estimate must be handed the same ones, or it weights a judgment by a bin it
    was not drawn from.

    **Serves the thinnest stratum first**, so a second batch buys resolution where the first
    left it thin rather than more of what is already well covered. With everything excluded
    from a previous batch, that is what makes "review 25 more" additive rather than repetitive.

    ``exclude_instance_ids`` are instances that must never be drawn: anything already judged,
    and anything that fed the method. Where a confirmed match becomes a new exemplar, that
    label *built* the method and cannot also score it, so the sample stays disjoint from it by
    construction. Passing the set explicitly rather than inferring it is what makes the
    constraint checkable rather than remembered.

    **By cluster, in proportion to size,** when ``stratum_of`` maps instances to clusters: each
    clip goes to the cluster furthest below its share (Sainte-Laguë, counting half a clip for the
    one it would take, which tracks each share more closely than counting a whole one). The draw
    serves the label's own figure; a cluster audited on its own is drawn from on its own, by the
    caller narrowing ``instances`` to it.
    """
    if budget <= 0:
        return []
    excluded = {str(i) for i in exclude_instance_ids}
    kind, groups = _strata(instances, stratum_of, edges)
    rng = np.random.default_rng(seed)
    pools: dict = {}
    covered: dict = {}
    for key, members in groups.items():
        available = sorted(
            (row for row in members if row.instance_id not in excluded),
            key=lambda r: r.instance_id,
        )
        # How much of this stratum a previous batch already took. Without it a second batch
        # repeats the first's allocation exactly — 25 across 10 strata is 3,3,3,3,3,2,2,2,2,2
        # both times, ending 6,6,6,6,6,4,4,4,4,4 — and "serves the thinnest first" is a
        # sentence in a docstring rather than something the code does.
        covered[key] = len(members) - len(available)
        if available:
            order = rng.permutation(len(available))
            pools[key] = [available[int(i)] for i in order]
    drawn: list = []
    used: dict = {key: 0 for key in pools}
    while len(drawn) < budget:
        candidates = [
            key for key in sorted(pools, key=str) if used[key] < len(pools[key])
        ]
        if not candidates:
            break
        if kind == "cluster":
            key = min(
                candidates,
                key=lambda k: (
                    (covered[k] + used[k] + 0.5) / max(len(groups[k]), 1),
                    str(k),
                ),
            )
        else:
            # Fewest already covered (previous batches plus this one), ties by stratum order so
            # the draw is deterministic rather than dict-ordered.
            key = min(candidates, key=lambda k: (covered[k] + used[k], k))
        drawn.append(pools[key][used[key]])
        used[key] += 1
    return drawn


def draw(
    run,
    *,
    label: str | None = None,
    path: str | None = None,
    budget: int | None = None,
    seed: int = 0,
    own_only: bool = False,
    node: int | None = None,
) -> list:
    """A batch drawn and logged as drawn, as ``AuthoredRun.audit_sample`` did: pending first,
    then a fresh stratified draw; the user's own clips never; a label edited since its audit
    draws its earlier clips again; a query run's bins recorded at its first draw."""
    budget = DEFAULT_BUDGET if budget is None else budget
    run_dir = run._run_dir()
    pool = run._audit_pool(label)
    if run.nested_labels:
        pool = run._audit_scope(pool, path, own_only=own_only)
    examples = run._audit_examples(pool)
    pool = [row for row in pool if row.instance_id not in examples]
    clusters = run._audit_clusters(pool)
    if node is not None:
        pool = [r for r in pool if (clusters or {}).get(r.instance_id) == node]
    since = None if node is not None else run._label_changed_at(path)
    decided = {
        row.instance_id
        for row in audit.latest_decisions(run_dir)
        if row.decided and (since is None or (row.drawn_at or "") >= since)
    }
    offered = {
        row.instance_id
        for row in audit.read_audit(run_dir)
        if since is None or (row.drawn_at or "") >= since
    }
    pending = [
        row
        for row in bo.bouts(pool)
        if row.instance_id in offered and row.instance_id not in decided
    ]
    drawn = pending[:budget]
    if len(drawn) < budget:
        # A scored run's draw recorded its bins (`_audit_edges(..., fix_seed=seed)` did); a
        # clustering run has none to record.
        edges = run._audit_edges({str(row.source_label) for row in pool})
        if edges is not None:
            edges = fix_bins(run_dir, edges, seed=seed)
        fresh = draw_sample(
            pool,
            stratum_of=clusters,
            budget=budget - len(drawn),
            seed=seed,
            exclude_instance_ids=decided | offered,
            edges=edges,
        )
        for row in fresh:
            audit.append_audit(run_dir, row.instance_id, score=row.score)
        drawn = [*drawn, *fresh]
    return drawn


def judge(run, instance_id: str, is_match: bool | None, *, answer: str = ""):
    """One decision, appended, as ``AuthoredRun.label_audit`` did."""
    return audit.append_audit(
        run._run_dir(), instance_id, human_label=is_match, decided=True, answer=answer
    )
