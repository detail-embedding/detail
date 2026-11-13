"""What one cluster is, in readable measures, and how far to trust that — the cluster profile
(DESIGN_PLAN §22 "Cluster profile").

The clip window's *What sets this cluster apart* ranks measures one at a time on the cluster's
most typical members (``api/clip_fingerprint.py``). A profile asks the deeper question: it
draws a **seeded random** sample of the node's members and an equal-sized comparison sample
(the rest of the run, the node's sibling, or another cluster named by id, which asks whether
two clusters in different branches are one behaviour), measures both with every registered descriptor
the data supports, and fits an explainer (``analysis/explainers.py``) scored on held-out
recordings beside a shuffled floor.

**Random, not typical.** Typical members are the easy ones; a classifier scored on them would
flatter itself. The draw is over the node cache's leaf runs (``NodeView.sample_runs``), seeded,
so the same profile is the same numbers tomorrow.

**Labels stay out.** The rows are told apart by cluster membership, which is the clustering's
own output. Hand labels, behavior labels and Query results are not read here at all: they may
one day score a suggested name, never build the profile.

**A time budget, not a count, by default.** Reading a track is the slow part, so the profile
reads animal-recordings in a seeded order until about half the budget is spent, measures every
sampled member in what it read, and balances the two sides. A fixed count per side skips the
budget. The saved profile records what was asked for and what was measured.

**Where and which way, measured and by default left out of the fit.** Position and facing
(``Descriptor.place``) are measured like everything else and kept in ``measures.csv``, but with
``leave_out_place`` (the default) the explainer never sees them: rearing toward the left and
toward the back are one behaviour to most people, and a profile that separated them on facing
alone would read as two.

**Two clusters compared: the place fit beside it.** A comparison against another cluster with
place left out also fits the same rows on position and facing *alone*
(``ClusterProfile.place_explanation``). Body measures that cannot tell two clusters apart while
place can is the signature of one behaviour seen in two places or directions — the evidence
look-alike groups rest on (``api/cluster_lookalikes.py``).

Saved beside the run it explains — ``analysis/<run>/profiles/node_<id>/profile.yaml`` and
``measures.csv`` — so reopening it is instant. A comparison against another cluster is saved in
its own folder under the node (``against_<other>/``), so it never replaces the node's own
profile. Nothing about the clustering itself is written.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from detail.analysis import explainers
from detail.core import yaml_load
from detail.core.utils import subject_key
from detail.configs.cluster_profile import (
    COMPARE_NODE,
    COMPARE_REST,
    COMPARE_SIBLING,
    ClusterProfileConfig,
)

SCHEMA_VERSION = 1
PROFILES_DIRNAME = "profiles"
PROFILE_FILENAME = "profile.yaml"
MEASURES_FILENAME = "measures.csv"

#: The most members drawn per side before the budget decides how many are read.
MAX_DRAW = 2000
#: The share of the budget spent reading and measuring; the rest is the fit.
MEASURE_SHARE = 0.6
#: Why a place measure is not in the fit, as a profile lists it.
LEFT_OUT_PLACE = "left out: position and facing"


@dataclass
class ContextClues:
    """What happens around the cluster: neighbours, bout length, when, and for whom."""

    before: list[dict] = field(default_factory=list)
    after: list[dict] = field(default_factory=list)
    bout_median_s: float = float("nan")
    bout_p10_s: float = float("nan")
    bout_p90_s: float = float("nan")
    session_thirds: list[float] = field(default_factory=list)
    over_represented: list[dict] = field(default_factory=list)


@dataclass
class ClusterProfile:
    run_id: str
    node: int
    unit: str
    compare: str
    comparison_node: int | None
    config: dict
    explanation: explainers.Explanation
    context: ContextClues
    basis: str
    labels: dict[str, str]
    units: dict[str, str]
    n_units_read: int
    elapsed_s: float
    created_at: str
    rows: list[dict] = field(default_factory=list)
    #: The same rows fitted on position and facing alone, for a comparison against another
    #: cluster with place left out; ``None`` otherwise.
    place_explanation: explainers.Explanation | None = None

    @property
    def headline(self) -> str:
        """One line a person reads first: whether the measures separate the cluster."""
        e = self.explanation
        if not np.isfinite(e.accuracy):
            return e.unscored or e.note or "not enough data to fit"
        verdict = (
            "The measures tell this cluster apart"
            if e.separated
            else "These measures don't separate this cluster"
        )
        return (
            f"{verdict}: {e.accuracy:.0%} balanced accuracy on held-out "
            f"{'recordings' if e.held_out == 'recording' else 'rows'} "
            f"(floor {e.floor:.0%}, margin {e.margin:.0%})."
        )


def place_text(profile: ClusterProfile) -> str:
    """What the place fit of a comparison with another cluster says, in one line, or ``""``
    without one. One rendering for the CLI and the profile window."""
    place, body = profile.place_explanation, profile.explanation
    if place is None:
        return ""
    if not np.isfinite(place.accuracy):
        why = place.unscored or place.note or "not enough data to fit"
        return f"Position and facing alone: {why}."
    line = (
        f"Position and facing alone: {place.accuracy:.0%} balanced accuracy "
        f"(floor {place.floor:.0%})"
    )
    if np.isfinite(body.accuracy) and not body.separated:
        if place.separated:
            return (
                f"{line} — the body does the same in both; they differ in where the "
                "animal is or which way it faces: one behaviour, seen twice."
            )
        return f"{line} — neither the body nor the place tells them apart."
    return f"{line}."


def comparison_of(
    view, node: int, compare: str, against: int | None = None
) -> tuple[int, int | None]:
    """``(node to draw from, node to leave out)`` for the comparison side.

    Raises:
        ValueError: For ``sibling`` on the root, which has none; for ``node`` without
            ``against``, outside the tree, or where one of the two holds the other.
    """
    node = int(node)
    if compare == COMPARE_REST:
        return view.root, node
    if compare == COMPARE_SIBLING:
        parents = view.ancestors(node)
        if not parents:
            raise ValueError(
                "the root has no sibling; compare against the rest instead"
            )
        others = [c for c in view.children(parents[-1]) if c != node]
        return int(others[0]), None
    if compare == COMPARE_NODE:
        if against is None:
            raise ValueError("compare 'node' needs the other cluster's node id")
        other = int(against)
        n_nodes = int(view.tree.n_nodes)
        if not 0 <= other < n_nodes:
            raise ValueError(
                f"node {other} is not in this run's tree (0 to {n_nodes - 1})"
            )
        if other == node:
            raise ValueError(f"cluster #{node} cannot be compared with itself")
        if other in view.ancestors(node):
            raise ValueError(
                f"cluster #{node} sits inside #{other}; compare it with its sibling or "
                "the rest of the run instead"
            )
        if node in view.ancestors(other):
            raise ValueError(
                f"cluster #{other} sits inside #{node}, so it is part of it; compare "
                f"#{other} with its sibling instead"
            )
        return other, None
    raise ValueError(
        f"unknown comparison {compare!r}; expected 'rest', 'sibling' or 'node'"
    )


def _basis(slices: list) -> str:
    """What the measures rest on, in words — the mask or only boxes, and which keypoint run
    and which of its three roles (nose, mid back, tailbase) were found."""
    from detail.analysis.descriptors import POINT_ALIASES, point_named

    if any(s.centroid is not None for s in slices):
        base = "mask (kinematics sidecar)"
    elif any(s.bbox is not None for s in slices):
        base = "boxes only (no kinematics sidecar; mask measures unavailable)"
    else:
        return "nothing readable"
    runs = sorted({s.points_run for s in slices if s.points_run})
    if not runs:
        return base
    with_points = [s for s in slices if s.points]
    roles = [
        role.replace("_", " ")
        for role in POINT_ALIASES
        if any(point_named(s.points, role) is not None for s in with_points)
    ]
    found = f" ({', '.join(roles)})" if roles else " (none of nose, mid back, tailbase)"
    return f"{base} + keypoints {', '.join(runs)}{found}"


def profile_cluster(
    experiment,
    run,
    node: int,
    config: ClusterProfileConfig | dict | None = None,
    *,
    progress=None,
) -> ClusterProfile:
    """Measure and explain ``node`` of clustering ``run`` against ``config.compare``.

    ``progress(stage, fraction, detail)`` hears each phase — ``sampling``, ``measuring``,
    ``fitting``, ``context`` — so a surface names what it is doing.

    Raises:
        ValueError: An unknown comparison, the root with ``sibling``, or an explainer whose
            modules are missing (the default falls back instead; see ``explainers.resolve``).
        FileNotFoundError: The run has no node cache yet.
    """
    from detail.analysis.descriptors import DESCRIPTORS, available, describe, in_scope
    from detail.api.frame_slices import between, track_slice
    from detail.core import run_selection

    config = ClusterProfileConfig.model_validate(config or {})
    explainers.resolve(config.explainer)  # refuse before reading anything
    began = time.monotonic()
    view = run.node_view()
    node = int(node)
    if progress is not None:
        progress("sampling", 0.0, "drawing members")
    draw_from, leave_out = comparison_of(view, node, config.compare, config.against)
    n_draw = config.n_per_side or MAX_DRAW
    cluster = view.sample_runs(node, n=n_draw, seed=config.seed)
    comparison = view.sample_runs(
        draw_from, n=n_draw, seed=config.seed + 1, outside=leave_out
    )
    by_unit: dict[tuple[str, int], list[tuple[dict, bool]]] = {}
    for spans, in_cluster in ((cluster, True), (comparison, False)):
        for span in spans:
            key = (span["recording_id"], subject_key(span["subject"]))
            by_unit.setdefault(key, []).append((span, in_cluster))
    unit_order = sorted(by_unit)
    np.random.default_rng(config.seed).shuffle(unit_order)
    embed_runs = run_selection.runs_by_recording(
        run.state(), {key[0] for key in unit_order}
    )

    names = list(DESCRIPTORS)
    rows: list[dict] = []
    read: list = []
    budget = config.time_budget_s * MEASURE_SHARE
    for index, key in enumerate(unit_order):
        elapsed = time.monotonic() - began
        if config.n_per_side is None and elapsed > budget and read:
            break
        if progress is not None:
            fraction = (
                index / len(unit_order)
                if config.n_per_side is not None
                else min(elapsed / budget, 1.0)
            )
            progress(
                "measuring",
                fraction,
                f"{experiment.recording_names().get(key[0], key[0])} · "
                f"{index + 1} of {len(unit_order)} animal-recordings",
            )
        trajectory = _trajectory(experiment, key, embed_runs.get(key[0]), track_slice)
        if trajectory is None or not len(trajectory.t_ns):
            continue
        read.append(trajectory)
        for span, in_cluster in by_unit[key]:
            start, end = span["start_t_ns"], span["end_t_ns"]
            bout = between(trajectory, start, end)
            present = set(available(bout)) if len(bout) else set()
            wanted = [
                n for n in names if n in present and in_scope(DESCRIPTORS[n], bout)
            ]
            values = describe(bout, names=wanted) if wanted else {}
            rows.append(
                {
                    "recording_id": key[0],
                    "subject": key[1],
                    "start_t_ns": int(span["start_t_ns"]),
                    "end_t_ns": int(span["end_t_ns"]),
                    "in_cluster": in_cluster,
                    **{n: float(values.get(n, np.nan)) for n in names},
                }
            )
    rows = _balanced(rows, config.seed)
    if progress is not None:
        progress("fitting", 0.0, f"{len(rows)} members measured")
    # Every measured name keeps its label and unit, skipped or not: a skipped one is listed by
    # what a person calls it, with why.
    labels = {n: DESCRIPTORS[n].label for n in names}
    units = {n: DESCRIPTORS[n].unit for n in names}
    # A measure that only means something from one camera view, on recordings that are not
    # that view, is named with the reason rather than reported as "missing on every row".
    view_reasons = {
        **_input_reasons(names, read, DESCRIPTORS),
        **_view_reasons(names, read, DESCRIPTORS),
    }
    if config.leave_out_place:
        view_reasons.update(
            {
                n: LEFT_OUT_PLACE
                for n in names
                if DESCRIPTORS[n].place and n not in view_reasons
            }
        )
    names = [n for n in names if n not in view_reasons]
    X = np.array([[r[n] for n in names] for r in rows], dtype=np.float64).reshape(
        len(rows), len(names)
    )
    y = np.array([r["in_cluster"] for r in rows], dtype=bool)
    groups = np.array([r["recording_id"] for r in rows], dtype=object)
    explanation = explainers.explain(
        X,
        y,
        groups,
        names,
        explainer=config.explainer,
        labels=labels,
        seed=config.seed,
        margin=config.margin,
    )
    explanation.measures_dropped = {**view_reasons, **explanation.measures_dropped}
    place_explanation = None
    if config.compare == COMPARE_NODE and config.leave_out_place:
        place_explanation = _place_fit(rows, labels, config, view_reasons)
    if progress is not None:
        progress("context", 0.0, "neighbours, bout length, groups")
    context = context_clues(experiment, view, node, run.node_names())
    return ClusterProfile(
        run_id=str(run.id),
        node=node,
        unit="behavior",
        compare=config.compare,
        comparison_node=int(draw_from) if leave_out is None else None,
        config=config.model_dump(),
        explanation=explanation,
        context=context,
        basis=_basis(read),
        labels=labels,
        units=units,
        n_units_read=len(read),
        elapsed_s=round(time.monotonic() - began, 2),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rows=rows,
        place_explanation=place_explanation,
    )


def _place_fit(rows, labels, config, view_reasons) -> explainers.Explanation | None:
    """The rows fitted on the place measures alone, or ``None`` when none was measured."""
    from detail.analysis.descriptors import DESCRIPTORS

    place = [
        n
        for n in labels
        if DESCRIPTORS[n].place
        and view_reasons.get(n, LEFT_OUT_PLACE) == LEFT_OUT_PLACE
    ]
    if not place or not rows:
        return None
    X = np.array([[r[n] for n in place] for r in rows], dtype=np.float64)
    y = np.array([r["in_cluster"] for r in rows], dtype=bool)
    groups = np.array([r["recording_id"] for r in rows], dtype=object)
    return explainers.explain(
        X,
        y,
        groups,
        place,
        explainer=config.explainer,
        labels={n: labels[n] for n in place},
        seed=config.seed,
        margin=config.margin,
    )


def _trajectory(experiment, key, embed_run_id, track_slice):
    """One animal's whole track from the best source its embedding run has, or ``None``."""
    recording_id, subject = key
    try:
        recording = experiment.get_recording(recording_id)
        record = next(
            (e for e in recording.embedding_runs if e.get("id") == embed_run_id), None
        )
        if record is None:
            return None
        return track_slice(recording, record, subject)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


#: What fills an input no read slice had, for the reason a profile gives.
_FILLED_BY = {
    "shape": "run `detail recording backfill-kinematics` to add the mask's outline",
    "motion": "run `detail recording backfill-motion` to measure movement inside the mask",
    "points": "run keypoint detect and keypoint track on the recording",
}


def _input_reasons(names: list[str], slices: list, descriptors: dict) -> dict[str, str]:
    """``{measure: why}`` for each measure whose input no read slice has, naming what fills
    it — rather than "missing on every row", which reads like a broken file."""
    out = {}
    for name in names:
        for need in descriptors[name].needs:
            if need in _FILLED_BY and not any(getattr(s, need, None) for s in slices):
                out[name] = f"no {need} on these recordings; {_FILLED_BY[need]}"
                break
    return out


def _view_reasons(names: list[str], slices: list, descriptors: dict) -> dict[str, str]:
    """``{measure: why}`` for each view-gated measure no read recording's camera view allows."""
    from detail.core import camera_view

    seen = {s.camera_view for s in slices}
    out = {}
    for name in names:
        views = descriptors[name].views
        if views is None or seen & set(views):
            continue
        wanted = " or ".join(camera_view.VIEWS[v].lower() for v in views)
        have = sorted(
            camera_view.VIEWS[v].lower() if v else "not set" for v in seen
        ) or ["not set"]
        out[name] = f"needs a {wanted} camera; the camera view is {', '.join(have)}"
    return out


def _balanced(rows: list[dict], seed: int) -> list[dict]:
    """Trim the larger side, at random, to the smaller one's count — a classifier scored on an
    imbalanced sample learns the imbalance."""
    rng = np.random.default_rng(seed + 2)
    mine = [r for r in rows if r["in_cluster"]]
    rest = [r for r in rows if not r["in_cluster"]]
    n = min(len(mine), len(rest))

    def take(side):
        if len(side) <= n:
            return side
        keep = sorted(rng.choice(len(side), size=n, replace=False).tolist())
        return [side[i] for i in keep]

    return take(mine) + take(rest)


def context_clues(experiment, view, node: int, node_names: dict) -> ContextClues:
    """Neighbours, bout length, time in session and group over-representation — read from the
    node cache and the recording information, no tracks."""
    before, after, _n_runs = view.neighbours(node)

    def ranked(counts: dict[int, int]) -> list[dict]:
        total = sum(counts.values())
        out = [
            {
                "node": int(b),
                "name": node_names.get(str(b)) or "",
                "share": round(c / total, 4) if total else 0.0,
            }
            for b, c in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
        return out[:3]

    durations = view.run_durations_s(node)
    clues = ContextClues(before=ranked(before), after=ranked(after))
    if len(durations):
        clues.bout_median_s = float(np.median(durations))
        clues.bout_p10_s = float(np.percentile(durations, 10))
        clues.bout_p90_s = float(np.percentile(durations, 90))
    clues.session_thirds = view.session_thirds(node)
    metadata = experiment.recording_metadata
    for column in experiment.metadata_columns:
        groups = {r: str(values.get(column, "")) for r, values in metadata.items()}
        if len({v for v in groups.values() if v}) < 2:
            continue
        distribution = view.distribution(node, groups)
        if distribution.flagged:
            clues.over_represented.append(
                {
                    "column": column,
                    "value": distribution.top,
                    "ratio": round(float(distribution.ratio), 2),
                    "floor": round(float(distribution.floor), 2),
                }
            )
    return clues


# --------------------------------------------------------------------------- #
# Saved profiles
# --------------------------------------------------------------------------- #
def profile_dir(run_dir: Path | str, node: int, against: int | None = None) -> Path:
    """``node``'s profile folder, or with ``against`` the folder of its comparison with that
    other cluster, under it."""
    out = Path(run_dir) / PROFILES_DIRNAME / f"node_{int(node)}"
    return out if against is None else out / f"against_{int(against)}"


def _against(profile: ClusterProfile) -> int | None:
    return profile.comparison_node if profile.compare == COMPARE_NODE else None


def _record(profile: ClusterProfile) -> dict:
    explanation = asdict(profile.explanation)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": profile.run_id,
        "node": profile.node,
        "unit": profile.unit,
        "compare": profile.compare,
        "comparison_node": profile.comparison_node,
        "params": profile.config,
        "basis": profile.basis,
        "labels": profile.labels,
        "units": profile.units,
        "n_units_read": profile.n_units_read,
        "elapsed_s": profile.elapsed_s,
        "created_at": profile.created_at,
        "explanation": _plain(explanation),
        "context": _plain(asdict(profile.context)),
        "place_explanation": (
            _plain(asdict(profile.place_explanation))
            if profile.place_explanation is not None
            else None
        ),
    }


def _plain(value):
    """YAML-safe: tuples to lists, numpy scalars to floats, NaN kept as ``.nan``."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_profile(run_dir: Path | str, profile: ClusterProfile) -> Path:
    """Write ``profile.yaml`` (``schema_version`` first) and ``measures.csv``, atomically."""
    from detail.core import atomic

    out = profile_dir(run_dir, profile.node, _against(profile))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (PROFILE_FILENAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(_record(profile), f, sort_keys=False, allow_unicode=True)
    atomic.replace(tmp, out / PROFILE_FILENAME)
    if profile.rows:
        columns = list(profile.rows[0])
        tmp = out / (MEASURES_FILENAME + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(profile.rows)
        atomic.replace(tmp, out / MEASURES_FILENAME)
    return out


def load_profile(
    run_dir: Path | str, node: int, against: int | None = None
) -> ClusterProfile | None:
    """The saved profile of ``node`` (or its comparison with cluster ``against``), or
    ``None`` when it was never made."""
    from detail.core import atomic

    path = profile_dir(run_dir, node, against) / PROFILE_FILENAME
    if not path.exists():
        return None
    record = yaml_load.safe_load(atomic.read_text(path))
    place = record.get("place_explanation")
    return ClusterProfile(
        run_id=record["run_id"],
        node=int(record["node"]),
        unit=record["unit"],
        compare=record["compare"],
        comparison_node=record.get("comparison_node"),
        config=record["params"],
        explanation=explanation_from(record["explanation"]),
        context=ContextClues(**record["context"]),
        basis=record["basis"],
        labels=record["labels"],
        units=record["units"],
        n_units_read=int(record["n_units_read"]),
        elapsed_s=float(record["elapsed_s"]),
        created_at=record["created_at"],
        place_explanation=explanation_from(place) if place else None,
    )


def explanation_from(e: dict) -> explainers.Explanation:
    """An ``Explanation`` as ``profile.yaml`` holds it, read back."""
    return explainers.Explanation(
        **{
            **e,
            "importances": [explainers.Importance(**i) for i in e["importances"]],
            "rules": [
                explainers.Rule(tuple(r["conditions"]), r["precision"], r["coverage"])
                for r in e["rules"]
            ],
        }
    )
