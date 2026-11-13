"""Look-alike clusters: which clusters in different branches of a run's tree are one behaviour,
and one label for them (§22 "Look-alike clusters").

A clustering tree splits one behaviour by where the animal is and which way it faces: rearing
toward the back and rearing toward the right land in different branches, sometimes far apart.
A depth cap cannot fix that (measured: a cap stops at nodes far too large); joining can. Two
clusters **look
alike** when a cluster profile of one against the other (``compare: node``) with position and
facing left out cannot tell them apart — what the body does is the same — while position and
facing alone usually can (``ClusterProfile.place_explanation``). Measured on one run, that
rule joins the rearing clusters from both halves of its tree.

- **Evidence is the saved comparisons** (``profiles/node_<a>/against_<b>/profile.yaml``), however
  they were made: the *Find look-alike clusters* job (``pipeline/cluster_lookalikes.py``) or one
  pair compared by hand in the Cluster profile window. Only comparisons with place left out
  count; a pair compared twice is read at its newest.
- **Groups chain.** A looks like B and B like C make one group of three, as the rearing clusters
  did. Each link is kept, so a person sees why a member is there, and can leave one out.
- **A group says what its labels would sweep in** (:func:`swept_clusters`). A label covers
  everything under its cluster with no label of its own, and clusters nest: on an_17 the
  rearing members held the walking clusters, and the outermost two were the tree's halves, so
  "the outer member covers the inner" proposed labelling 75% of the run one name. Every member
  is listed; one that would also label a cluster outside the group says which, and
  starts unticked.
- **A group is a suggestion, never a label.** Its proposed name is the label most of its members
  already carry, else none: the person gives one. *Label as one* goes through the label edit
  every label change uses (``LabelOperations._label_edit``), refused by ``check_labels`` as a
  typed name would be, and records where each label came from (``node_label_sources``, ``from:
  lookalikes``).
- **Which pairs to test** (:func:`candidate_pairs`): every pair of **branch tops** — the two
  halves of every split where both are at least the run's minimum cluster time. Pairs inside one
  another are never asked: a profile refuses them. ``pairs: labelled`` keeps only the pairs with
  a labelled side (:func:`labelled_pairs`), far fewer; :func:`count_pairs` says how many each
  mode compares, and about how long, for the warning every surface shows before every pair.

Reading is O(compared pairs) small files, each parsed once per change of its mtime
(``read_cache.cached_read``), so never on a UI thread.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

MODULE_ID = "cluster_lookalikes"
#: What ``node_label_sources[node]["from"]`` says for a label given as one to a group.
LOOKALIKES_SOURCE = "lookalikes"
#: The job's steps; the head of every message its runner emits is one of these.
PHASES = ("comparing",)


# --------------------------------------------------------------------------- #
# Which pairs
# --------------------------------------------------------------------------- #
def branch_tops(view) -> list[int]:
    """The two halves of every split where both are at least the run's minimum cluster time.

    A split where one side is under the floor only trims a sliver off a cluster, which stays
    the same behaviour; the halves of a real split are the clusters a person names.
    """
    floor_ns = float(view.min_cluster_s) * 1e9
    over = {
        node
        for node in range(int(view.tree.n_nodes))
        if view.summary(node).time_ns >= floor_ns
    }
    tops = set()
    for node in over:
        children = list(view.children(node))
        if len(children) == 2 and all(c in over for c in children):
            tops.update(int(c) for c in children)
    return sorted(tops)


def candidate_pairs(view) -> list[tuple[int, int]]:
    """``(a, b)`` for each pair of branch tops to compare, ``a < b``, in that order. Neither
    holds the other."""
    tops = branch_tops(view)
    ancestors = {n: set(view.ancestors(n)) for n in tops}
    return [
        (a, b)
        for i, a in enumerate(tops)
        for b in tops[i + 1 :]
        if a not in ancestors[b] and b not in ancestors[a]
    ]


def covering_label(tree, node: int, node_labels: dict) -> str | None:
    """The label id covering ``node``: its own, else the nearest labelled branch above it (the
    most specific label wins, as everywhere a label is read), else ``None``."""
    from detail.api.authored_run import parent_label

    return node_labels.get(str(int(node))) or parent_label(tree, int(node), node_labels)


def labelled_pairs(view, node_labels: dict) -> list[tuple[int, int]]:
    """The :func:`candidate_pairs` with at least one side a label covers, less those whose two
    sides one label covers (already one behaviour). ``node_labels`` is the run's ``{node id:
    label id}``: each labelled branch top against every other branch top."""
    cover: dict[int, str | None] = {}

    def label(node: int) -> str | None:
        if node not in cover:
            cover[node] = covering_label(view.tree, node, node_labels)
        return cover[node]

    return [
        (a, b)
        for a, b in candidate_pairs(view)
        if (label(a) or label(b)) and label(a) != label(b)
    ]


def labelled_tops(view, node_labels: dict) -> list[int]:
    """The branch tops a label covers."""
    return [
        top
        for top in branch_tops(view)
        if covering_label(view.tree, top, node_labels) is not None
    ]


def pairs_for(view, node_labels: dict, mode: str) -> list[tuple[int, int]]:
    """The pairs ``mode`` (``ClusterLookalikesConfig.pairs``) compares."""
    from detail.configs.cluster_lookalikes import PAIRS_ALL, PAIRS_LABELLED

    if mode == PAIRS_ALL:
        return candidate_pairs(view)
    if mode == PAIRS_LABELLED:
        return labelled_pairs(view, node_labels)
    raise ValueError(f"unknown pairs {mode!r}; expected 'all' or 'labelled'")


# --------------------------------------------------------------------------- #
# How many pairs, and about how long
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairCounts:
    """How many pairs each mode compares on one run, and about how long on this machine.

    ``n_all`` / ``n_labelled`` the pairs ``pairs: all`` / ``pairs: labelled`` resolve to;
    ``skipped_all`` / ``skipped_labelled`` how many of them were already compared with these
    settings (skipped while ``skip_existing`` is on); ``n_labelled_tops`` the branch tops a label
    covers. ``per_pair_s`` what one pair should take, ``basis`` what that rests on.
    """

    n_all: int
    n_labelled: int
    n_labelled_tops: int
    skipped_all: int
    skipped_labelled: int
    per_pair_s: float
    basis: str

    @property
    def counts(self) -> dict[str, int]:
        """``{"all": n, "labelled": n}``."""
        return {"all": self.n_all, "labelled": self.n_labelled}

    def _pairs(self, mode: str) -> tuple[int, int]:
        from detail.configs.cluster_lookalikes import PAIRS_ALL

        if mode == PAIRS_ALL:
            return self.n_all, self.skipped_all
        return self.n_labelled, self.skipped_labelled

    def to_compare(self, mode: str) -> int:
        """The pairs ``mode`` would compare: those it resolves to, less the skipped."""
        n, skipped = self._pairs(mode)
        return n - skipped

    def seconds(self, mode: str) -> float:
        """About how long ``mode`` takes here: the pairs it would compare, not the skipped."""
        return self.to_compare(mode) * self.per_pair_s

    def _amount(self, mode: str) -> str:
        from detail.core.progress import format_duration

        n, skipped = self._pairs(mode)
        text = f"{n:,} pair" + ("s" if n != 1 else "")
        if skipped:
            text = f"{n - skipped:,} of {text} left ({skipped:,} already compared)"
        return f"{text}, about {format_duration(self.seconds(mode))} on this machine"

    def warning(self) -> str:
        """What every surface says before comparing every pair (when any is left to
        compare: :meth:`to_compare`)."""
        from detail.configs.cluster_lookalikes import PAIRS_ALL, PAIRS_LABELLED

        lines = [
            f"Comparing every pair is compute intensive: {self._amount(PAIRS_ALL)}.",
            f"One pair takes {self.basis}.",
        ]
        if self.n_labelled_tops:
            lines.append(
                "Labelled pairs only ('labelled', each labelled cluster against every "
                f"other) compares fewer: {self._amount(PAIRS_LABELLED)}."
            )
        else:
            lines.append(
                "Labelled pairs only ('labelled') compares fewer, but no cluster is "
                "labelled yet: label at least one first."
            )
        return "\n".join(lines)


def pair_seconds(
    evidence, wanted: dict, time_budget_s: float, fixed: bool
) -> tuple[float, str]:
    """``(seconds, basis)`` for one pair: the median time of this run's pairs already compared
    with ``wanted``'s settings, else the time budget (as ``profile_describe.profile_seconds``
    takes a profile it has not timed)."""
    import statistics

    from detail.api.profile_describe import PROFILE_OVERHEAD_S
    from detail.core.progress import format_duration

    taken = [
        item.elapsed_s
        for item in evidence
        if math.isfinite(item.elapsed_s) and same_settings(item.params, wanted)
    ]
    if taken:
        median = statistics.median(taken)
        return median, (
            f"about {format_duration(median)} (the median of {len(taken)} pair"
            + ("s" if len(taken) != 1 else "")
            + " this run already has)"
        )
    budget = float(time_budget_s) + PROFILE_OVERHEAD_S
    if fixed:
        return budget, (
            f"about {format_duration(budget)} (a fixed count is not timed in advance; "
            "taken at the time budget)"
        )
    return budget, f"about {format_duration(budget)} (the time budget)"


def count_pairs(view, node_labels: dict, evidence, config) -> PairCounts:
    """:class:`PairCounts` for a ``ClusterLookalikesConfig`` ``config`` over ``view``.
    ``evidence`` is the run's :func:`read_evidence`."""
    from detail.configs.cluster_lookalikes import PROFILE_FIELDS

    wanted = {name: getattr(config, name) for name in PROFILE_FIELDS}
    every = candidate_pairs(view)
    labelled = labelled_pairs(view, node_labels)
    done = (
        {item.pair for item in evidence if same_settings(item.params, wanted)}
        if config.skip_existing
        else set()
    )
    per_pair, basis = pair_seconds(
        evidence, wanted, config.time_budget_s, config.n_per_side is not None
    )
    return PairCounts(
        n_all=len(every),
        n_labelled=len(labelled),
        n_labelled_tops=len(labelled_tops(view, node_labels)),
        skipped_all=sum(1 for p in every if p in done),
        skipped_labelled=sum(1 for p in labelled if p in done),
        per_pair_s=per_pair,
        basis=basis,
    )


# --------------------------------------------------------------------------- #
# The evidence: saved comparisons of one cluster against another
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairEvidence:
    """One saved comparison of cluster ``node`` against cluster ``against``.

    ``accuracy`` / ``floor`` are the body-measure fit's held-out balanced accuracy and shuffled
    floor; ``place_accuracy`` / ``place_floor`` the same rows fitted on position and facing
    alone (NaN for a comparison saved before that fit existed). ``file`` is the profile's path
    relative to the run, POSIX-separated.
    """

    node: int
    against: int
    accuracy: float
    floor: float
    margin: float
    separated: bool
    place_accuracy: float
    place_floor: float
    params: dict
    created_at: str
    file: str
    #: How long the comparison took (NaN for a file without it).
    elapsed_s: float = float("nan")

    @property
    def pair(self) -> tuple[int, int]:
        return tuple(sorted((self.node, self.against)))

    @property
    def looks_alike(self) -> bool:
        """Body measures could be fitted and could not tell the two apart."""
        return math.isfinite(self.accuracy) and not self.separated

    @property
    def lift(self) -> float:
        return self.accuracy - self.floor

    @property
    def place_lift(self) -> float:
        return self.place_accuracy - self.place_floor

    def text(self) -> str:
        """``#755 vs #1067: body measures 52% (floor 50%) · position and facing alone 91%``."""
        body = (
            f"body measures {self.accuracy:.0%} (floor {self.floor:.0%})"
            if math.isfinite(self.accuracy)
            else "body measures could not be fitted"
        )
        place = (
            f" · position and facing alone {self.place_accuracy:.0%}"
            if math.isfinite(self.place_accuracy)
            else ""
        )
        return f"#{self.node} vs #{self.against}: {body}{place}"


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _evidence(path: Path, run_dir: Path) -> PairEvidence | None:
    """One comparison file read, or ``None`` when it is not a place-left-out comparison of two
    clusters or cannot be read."""
    from detail.configs.cluster_profile import COMPARE_NODE
    from detail.core import atomic
    from detail.core import yaml_load

    try:
        record = yaml_load.safe_load(atomic.read_text(path)) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(record, dict) or record.get("compare") != COMPARE_NODE:
        return None
    params = record.get("params") or {}
    if not params.get("leave_out_place", True):
        return None
    if record.get("comparison_node") is None:
        return None
    e = record.get("explanation") or {}
    place = record.get("place_explanation") or {}
    return PairEvidence(
        node=int(record["node"]),
        against=int(record["comparison_node"]),
        accuracy=_float(e.get("accuracy")),
        floor=_float(e.get("floor")),
        margin=_float(e.get("margin")),
        separated=bool(e.get("separated")),
        place_accuracy=_float(place.get("accuracy")),
        place_floor=_float(place.get("floor")),
        params=dict(params),
        created_at=str(record.get("created_at") or ""),
        file=path.relative_to(run_dir).as_posix(),
        elapsed_s=_float(record.get("elapsed_s")),
    )


def _evidence_cached(path: Path, run_dir: Path) -> PairEvidence | None:
    from detail.storage import read_cache

    return read_cache.cached_read(
        ("lookalike-evidence", str(path)), path, lambda: _evidence(path, run_dir)
    )


def _scan(folder: Path, prefix: str) -> list[tuple[int, Path]]:
    try:
        entries = list(os.scandir(folder))
    except (FileNotFoundError, NotADirectoryError):
        return []
    n = len(prefix)
    return [
        (int(entry.name[n:]), Path(entry.path))
        for entry in entries
        if entry.name.startswith(prefix) and entry.name[n:].isdigit()
    ]


def read_evidence(run_dir: Path | str) -> list[PairEvidence]:
    """Every saved comparison of one cluster against another with place left out, all of
    them (a pair compared twice or both ways is listed each time). O(compared pairs) files:
    never on the UI thread."""
    from detail.api.cluster_profile import PROFILE_FILENAME, PROFILES_DIRNAME

    run_dir = Path(run_dir)
    out = []
    for _node, folder in _scan(run_dir / PROFILES_DIRNAME, "node_"):
        for _other, against in _scan(folder, "against_"):
            evidence = _evidence_cached(against / PROFILE_FILENAME, run_dir)
            if evidence is not None:
                out.append(evidence)
    return out


def newest_per_pair(evidence) -> dict[tuple[int, int], PairEvidence]:
    """``{(a, b): its newest comparison}``, ``a < b``, whichever way round it was asked."""
    out: dict[tuple[int, int], PairEvidence] = {}
    for item in evidence:
        held = out.get(item.pair)
        if held is None or item.created_at > held.created_at:
            out[item.pair] = item
    return out


def same_settings(params: dict, wanted: dict) -> bool:
    """Whether a saved comparison's ``params`` were made with the job's profile settings
    ``wanted`` (``ClusterLookalikesConfig``'s ``PROFILE_FIELDS``). A field it predates reads
    at its default."""
    from detail.configs.cluster_profile import ClusterProfileConfig

    try:
        made = ClusterProfileConfig.model_validate(params or {})
    except ValueError:
        return False
    return all(getattr(made, k) == v for k, v in wanted.items())


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweptCluster:
    """A cluster a label on a group member would also cover, outside the group.

    Inside the member, not a member, and under no label of its own. ``share`` is the part of the
    run the label would take through it: its time less what members and labelled clusters
    inside it keep.
    """

    node: int
    share: float

    def text(self) -> str:
        """``#739 (3.9% of the run)``."""
        return f"#{self.node} ({self.share:.1%} of the run)"


@dataclass(frozen=True)
class LookalikeGroup:
    """Clusters that look alike, chained through the pairs that do.

    ``nodes`` every member, largest first. ``links`` the look-alike comparisons holding them
    together. ``sweeps`` ``{member: the clusters outside the group its label would also
    cover}`` (:class:`SweptCluster`), for each member that has any: a label covers everything
    under its cluster with no label of its own, and where clusters nest, a look-alike member
    can hold another behaviour. ``ticked`` the members to label by default: those that sweep
    nothing in, less any inside another of them (whose label already covers it). ``name`` the
    proposed label, ``named`` how it was chosen. ``labels`` / ``time_s`` per member; ``run_s``
    the whole run's time.
    """

    nodes: tuple[int, ...]
    links: tuple[PairEvidence, ...]
    sweeps: dict
    ticked: tuple[int, ...]
    name: str
    named: str
    labels: dict
    time_s: dict
    run_s: float

    def share(self, node: int) -> float:
        """A member's share of the run, 0-1."""
        return self.time_s[node] / self.run_s if self.run_s else float("nan")

    def text(self, unit: str = "cluster") -> str:
        members = ", ".join(f"#{n}" for n in self.nodes)
        name = f"“{self.name}”" if self.name else "no name yet"
        return f"{len(self.nodes)} {unit}s look alike ({members}) — {name}"


def _components(pairs) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    out: dict[int, set[int]] = {}
    for x in list(parent):
        out.setdefault(find(x), set()).add(x)
    return list(out.values())


def proposed_name(nodes, labels: dict) -> tuple[str, str]:
    """``(name, how it was chosen)`` for a group, ``nodes`` largest first: the label its
    members carry most (a person's choice wins), the larger cluster's on a tie; ``("", ...)``
    when none has one."""
    names = [str(labels[n]).strip() for n in nodes if str(labels.get(n) or "").strip()]
    if not names:
        return "", "no member has a label"
    counts = Counter(name.casefold() for name in names)
    best = max(counts.values())
    # `names` is in member order (largest first), so the first that ties is the largest's.
    name = next(name for name in names if counts[name.casefold()] == best)
    return name, (
        f"a label {best} of them already carry" if best > 1 else "a member's label"
    )


def find_groups(view, evidence, labels: dict | None = None) -> list[LookalikeGroup]:
    """The look-alike groups in ``evidence`` (:func:`read_evidence`), the most clusters first.

    ``labels`` ``{node: label}`` name the members, propose the group's name and decide what
    each member's label would sweep in (:func:`swept_clusters`).
    """
    labels = {int(k): str(v) for k, v in (labels or {}).items() if str(v or "")}
    by_pair = newest_per_pair(evidence)
    alike = [pair for pair, item in by_pair.items() if item.looks_alike]
    if not alike:
        return []
    tops = branch_tops(view)
    run_s = view.summary(view.root).time_ns / 1e9
    groups = []
    for component in _components(alike):
        time_s = {n: view.summary(n).time_ns / 1e9 for n in component}
        members = sorted(component, key=lambda n: (-time_s[n], n))
        sweeps = {
            m: swept
            for m in members
            if (swept := swept_clusters(view, m, component, tops, labels))
        }
        clean = [m for m in members if m not in sweeps]
        ticked = tuple(
            m for m in clean if not any(a in clean for a in view.ancestors(m))
        )
        name, named = proposed_name(members, labels)
        links = tuple(
            sorted(
                (
                    item
                    for pair, item in by_pair.items()
                    if item.looks_alike and set(pair) <= component
                ),
                key=lambda item: item.pair,
            )
        )
        groups.append(
            LookalikeGroup(
                nodes=tuple(members),
                links=links,
                sweeps=sweeps,
                ticked=ticked,
                name=name,
                named=named,
                labels={n: labels[n] for n in component if n in labels},
                time_s=time_s,
                run_s=run_s,
            )
        )
    return sorted(groups, key=lambda g: (-len(g.nodes), -max(g.time_s.values())))


def swept_clusters(
    view, member: int, group, tops, labels: dict
) -> tuple[SweptCluster, ...]:
    """The branch tops a label on ``member`` would also cover that are not in the group,
    outermost first, the largest first.

    A label covers everything under its cluster that has no label of its own (the most
    specific label wins), so on a tree whose clusters nest, a member can hold clusters that
    are another behaviour: an_17's rearing clusters held its walking ones, and the outermost
    two members were the tree's halves. A top inside ``member`` is swept unless it is in
    ``group``, or carries a label or sits under one nearer than ``member``. A top inside another
    listed one, with no member or label between, is said through it.
    """
    group = set(group)
    found = []
    for top in tops:
        if top in group or top in labels:
            continue
        above = view.ancestors(top)
        if member not in above:
            continue
        between = above[above.index(member) + 1 :]
        if any(a in labels for a in between):
            continue
        found.append(top)
    kept = group | set(labels)
    root_ns = view.summary(view.root).time_ns or 1
    out = []
    for top in found:
        # Said through a listed top above it only when nothing kept lies between: a listed
        # top's share leaves out the members inside it, and with them what they hold.
        for holder in reversed(view.ancestors(top)):
            if holder == member or holder in kept or holder in found:
                break
        if holder in found:
            continue
        inner = [k for k in kept if top in view.ancestors(k)]
        maximal = [k for k in inner if not any(a in inner for a in view.ancestors(k))]
        time_ns = view.summary(top).time_ns - sum(
            view.summary(k).time_ns for k in maximal
        )
        out.append(SweptCluster(int(top), time_ns / root_ns))
    return tuple(sorted(out, key=lambda s: (-s.share, s.node)))


# --------------------------------------------------------------------------- #
# One label for a group
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AcceptStep:
    """What labelling one node does — or why it is skipped.

    ``joins`` is the label it joins (as that label is spelled) and ``joins_clusters`` how many
    clusters carry it before this one. ``joins_yours`` is a label that was there before this
    plan, rather than one made by an earlier step of the same plan. ``file`` is the source file
    a label came from; a group's plan leaves it empty.
    """

    node: int
    name: str
    file: str
    accepted: bool
    joins: str = ""
    joins_clusters: int = 0
    joins_yours: bool = False
    reason: str = ""

    def text(self, unit: str = "cluster") -> str:
        """One line of the summary a person reads before anything is written."""
        head = f"{unit} #{self.node}: “{self.name}”"
        if not self.accepted:
            return f"{head} — skipped: {self.reason}"
        if not self.joins:
            return f"{head} — a new label"
        whose = "your label" if self.joins_yours else "the label"
        plural = "s" if self.joins_clusters != 1 else ""
        return f"{head} — joins {whose} “{self.joins}” ({self.joins_clusters} {unit}{plural})"


@dataclass(frozen=True)
class AcceptPlan:
    """Every step of a plan, in the order it applies: parents before their children."""

    steps: tuple[AcceptStep, ...]

    @property
    def accepted(self) -> tuple[AcceptStep, ...]:
        return tuple(step for step in self.steps if step.accepted)

    @property
    def skipped(self) -> tuple[AcceptStep, ...]:
        return tuple(step for step in self.steps if not step.accepted)

    def text(self, unit: str = "cluster") -> str:
        """The whole summary: what will be labelled, then what is skipped and why."""
        units = f"{unit}s"
        lines = [
            f"Will label {len(self.accepted)} {unit if len(self.accepted) == 1 else units}"
            + (":" if self.accepted else ".")
        ]
        lines += [f"  {step.text(unit)}" for step in self.accepted]
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)}:")
            lines += [f"  {step.text(unit)}" for step in self.skipped]
        return "\n".join(lines)


def _parents(parent) -> list[int]:
    return [int(p) for p in parent]


def _depth(parents: list[int], node: int) -> int:
    depth, current = 0, int(parents[int(node)])
    while current >= 0:
        depth += 1
        current = parents[current]
    return depth


def plan_join(names: dict, parent, nodes, name: str) -> AcceptPlan:
    """What labelling ``nodes`` as one label ``name`` would do to labels ``names``.

    Each node in turn, parents before children, against the labels as the steps before it left
    them: skipped when it already carries that label, when it has another label of its own
    (a person's label is kept), or when ``check_labels`` refuses the result — with the rule's
    own message. Pure: the GUI plans from what it shows, and the API plans again inside the
    edit.
    """
    from types import SimpleNamespace

    from detail.api.authored_run import check_labels
    from detail.storage import run_labels

    name = str(name or "").strip()
    parents = _parents(parent)
    tree = SimpleNamespace(parent=parents)
    current = {str(k): str(v) for k, v in names.items() if str(v or "").strip()}
    before = {v.strip().casefold() for v in current.values()}
    key = name.casefold()
    steps = []
    for node in sorted({int(n) for n in nodes}, key=lambda n: (_depth(parents, n), n)):

        def skip(reason: str, _node=node) -> None:
            steps.append(AcceptStep(_node, name, "", False, reason=reason))

        if not name:
            skip("no name given")
            continue
        own = current.get(str(node))
        if own and own.strip().casefold() == key:
            skip(f"it already has the label “{own}”")
            continue
        if own:
            skip(f"it already has your label “{own}”, which is kept")
            continue
        joined = [v for v in current.values() if v.strip().casefold() == key]
        trial = {**current, str(node): joined[0] if joined else name}
        labels, node_labels = run_labels.from_node_names(trial)
        try:
            check_labels(labels, node_labels, tree)
        except ValueError as exc:
            skip(str(exc))
            continue
        steps.append(
            AcceptStep(
                node,
                name,
                "",
                True,
                joins=joined[0] if joined else "",
                joins_clusters=len(joined),
                joins_yours=bool(joined) and key in before,
            )
        )
        current = trial
    return AcceptPlan(tuple(steps))


def plan_lines(
    plan: AcceptPlan, unit: str = "cluster", sweeps: dict | None = None
) -> tuple[list[str], list[str]]:
    """``(what is labelled, what is skipped and why)``, one line each, for a group's plan.

    The steps of one group label one label, so a step joining the label an earlier step made
    is said as part of it; only a label the person already had is said to be joined. A step
    whose cluster sweeps others in (``sweeps``, :attr:`LookalikeGroup.sweeps`) names them.
    """
    sweeps = sweeps or {}
    labelled = []
    for step in plan.accepted:
        line = f"{unit} #{step.node}"
        if step.joins_yours:
            plural = "s" if step.joins_clusters != 1 else ""
            line += f" — joins your label “{step.joins}” ({step.joins_clusters} {unit}{plural})"
        if sweeps.get(step.node):
            line += " — also labels " + ", ".join(s.text() for s in sweeps[step.node])
        labelled.append(line)
    return labelled, [step.text(unit) for step in plan.skipped]


def sweeps_in(plan: AcceptPlan, sweeps: dict | None) -> bool:
    """Whether a step the plan accepts would label a cluster outside the group."""
    return any((sweeps or {}).get(step.node) for step in plan.accepted)


def plan_text(
    plan: AcceptPlan, name: str, unit: str = "cluster", sweeps: dict | None = None
) -> str:
    """The whole summary of a group's plan, for a person to read before anything is written."""
    labelled, skipped = plan_lines(plan, unit, sweeps)
    n = len(labelled)
    lines = [
        f"Will label {n} {unit if n == 1 else unit + 's'} “{name}”"
        + (":" if n else ".")
    ]
    lines += [f"  {line}" for line in labelled]
    if skipped:
        lines.append(f"Skipped {len(skipped)}:")
        lines += [f"  {line}" for line in skipped]
    if sweeps_in(plan, sweeps):
        lines.append(SWEEP_WARNING.format(unit=unit))
    return "\n".join(lines)


#: Said wherever a label would cover clusters outside the group.
SWEEP_WARNING = (
    "A label covers everything under its {unit} that has no label of its own, so the {unit}s "
    "named above take it too. Label them first, or leave out the {unit} that holds them."
)


class Lookalikes:
    """The look-alike layer on a clustering run (§22). Pose and temporal share it.

    The host supplies ``_run_dir``, ``_tree``, ``node_view``, ``state`` and ``_label_edit``
    (``LabelOperations``).
    """

    def lookalike_groups(self) -> list[LookalikeGroup]:
        """The run's look-alike groups from every saved comparison of two clusters, named
        from its labels. Reads O(compared pairs) files: never on a UI thread."""
        return find_groups(
            self.node_view(),
            read_evidence(self._run_dir()),
            {int(k): v for k, v in self.state().node_names.items()},
        )

    def plan_lookalike_label(self, nodes, name: str) -> AcceptPlan:
        """What :meth:`label_lookalikes` would do, writing nothing."""
        return plan_join(self.state().node_names, self._tree().parent, nodes, name)

    def label_lookalikes(self, nodes, name: str, *, progress=None) -> AcceptPlan:
        """Give ``nodes`` one label, ``name``, through ``name_nodes``' own edit (§22).

        A name that exists joins that label; a node with another label of its own keeps it; a
        node the label rules refuse is skipped with the rule's reason. One edit for the whole
        group — one check, one history entry — and nothing is written when no step is
        accepted. Each label made this way records in ``node_label_sources`` that it came from
        a look-alike group, and which.

        Args:
            nodes: The clusters to label as one.
            name: The label's name.
            progress: The re-projection's ``(stage, fraction, detail)`` callback.

        Returns:
            The plan as applied: what was labelled and what was skipped, with why.

        Raises:
            ReadOnlyRunError: This is a snapshot view.
        """
        from detail.api.authored_run import apply_names
        from detail.core.utils import utc_timestamp

        parent = self._tree().parent
        group = sorted({int(n) for n in nodes})
        planned = [plan_join(self.state().node_names, parent, group, name)]
        if not planned[0].accepted:
            return planned[0]

        def change(state) -> dict:
            plan = plan_join(state.node_names, parent, group, name)
            planned.append(plan)
            apply_names(state, {str(step.node): step.name for step in plan.accepted})
            now = utc_timestamp()
            for step in plan.accepted:
                state.node_label_sources[str(step.node)] = {
                    "from": LOOKALIKES_SOURCE,
                    "group": group,
                    "label": state.node_labels[str(step.node)],
                    "accepted_at": now,
                }
            return {
                "group": group,
                "name": name,
                "labelled": [step.node for step in plan.accepted],
                "skipped": {str(step.node): step.reason for step in plan.skipped},
            }

        self._label_edit("label look-alikes", change, progress=progress)
        return planned[-1]
