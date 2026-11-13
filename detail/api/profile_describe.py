"""Profile many clusters of a clustering run — the pieces of the batch job (§22 "Profile
clusters"; module id ``profile_describe``, kept so the runs made before it lost its description
step still open).

The job itself is an analysis module (``pipeline/profile_describe.py``): one executor run, its
own ``an_NN`` record listing each cluster's status, progress through the shared event shape,
cancel and stop through the executor. What it does per cluster is exactly what the single-cluster
surfaces do — ``Experiment.profile_cluster`` — written into the job's own run directory and then
placed (:func:`place`) where that saves it (``profiles/node_<id>/`` beside the clustering run),
so the Interpret tab shows them as if made one at a time.

What lives here is what the job, the API, the CLI and the GUI share:

- **Which clusters** (:func:`select_nodes`): the nodes at or above the run's minimum cluster
  time (``min_cluster_s``, the floor its clips are built at), every node
  with members, the leaves, or a list. Never the root, which has nothing to be compared with.
- **What is already done** (:func:`profile_matches`): a profile made with the same settings.
- **How long it should take** (:func:`estimate`): each profile at its time budget, or the
  median this run's own profiles with these settings took.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from detail.configs.cluster_profile import ClusterProfileConfig
from detail.configs.profile_describe import (
    CLUSTERS_ALL,
    CLUSTERS_FLOOR,
    CLUSTERS_IDS,
    CLUSTERS_LEAVES,
    PROFILE_FIELDS,
    ProfileDescribeConfig,
)
from detail.core.progress import format_duration

MODULE_ID = "profile_describe"

#: The job's steps, in the order each cluster moves through them; the head of every message the
#: runner emits is one of these (``core/progress.message_phase``). ``measuring`` reads the
#: members' tracks (most of the time); ``fitting`` fits the explainer and adds the context.
PHASES = ("measuring", "fitting")

#: What a profile is taken to cost beside its time budget, seconds (the context clues and the
#: save).
PROFILE_OVERHEAD_S = 2.0


# --------------------------------------------------------------------------- #
# Which clusters
# --------------------------------------------------------------------------- #
def select_nodes(view, clusters: str, node_ids: list[int] | None = None) -> list[int]:
    """The nodes ``clusters`` names in ``view``'s tree, largest (by member time) first, so a
    run stopped part-way has done the clusters that matter most; an explicit list keeps its
    own order.

    Raises:
        ValueError: An id outside the tree, the root asked for by id, an unknown choice, or
            ``ids`` with no ids.
    """
    root = view.root
    n_nodes = int(view.tree.n_nodes)
    if clusters == CLUSTERS_IDS:
        ids = [int(n) for n in node_ids or []]
        if not ids:
            raise ValueError("clusters 'ids' needs at least one node id")
        outside = [n for n in ids if not 0 <= n < n_nodes]
        if outside:
            raise ValueError(
                f"node ids {outside} are not in this run's tree (0 to {n_nodes - 1})"
            )
        if root in ids:
            raise ValueError(
                f"node {root} is the root: it holds every member, so there is nothing to "
                "compare it against; leave it out"
            )
        return list(dict.fromkeys(ids))
    if clusters not in (CLUSTERS_FLOOR, CLUSTERS_ALL, CLUSTERS_LEAVES):
        raise ValueError(
            f"unknown clusters {clusters!r}; expected floor, all, leaves or ids"
        )
    floor_ns = float(view.min_cluster_s) * 1e9
    chosen = []
    for node in range(n_nodes):
        if node == root:
            continue
        summary = view.summary(node)
        if summary.members <= 0:
            continue
        # The floor is the run's own minimum cluster time, not whether the cache happened to
        # list the node: a cache built without member lists (an_10, 2026-10-22) lists none.
        if clusters == CLUSTERS_FLOOR and summary.time_ns < floor_ns:
            continue
        if clusters == CLUSTERS_LEAVES and node >= view.tree.n_leaves:
            continue
        chosen.append((-int(summary.time_ns), node))
    return [node for _time, node in sorted(chosen)]


# --------------------------------------------------------------------------- #
# What is already done
# --------------------------------------------------------------------------- #
def profile_config(config: ProfileDescribeConfig) -> ClusterProfileConfig:
    """The per-cluster profile settings of a batch config, carried by name."""
    return ClusterProfileConfig(**{f: getattr(config, f) for f in PROFILE_FIELDS})


def profile_matches(saved, wanted: ClusterProfileConfig) -> bool:
    """Whether ``saved`` (a ``ClusterProfile`` or ``None``) was made with ``wanted``'s
    settings. A field a saved profile predates reads at its default."""
    if saved is None:
        return False
    try:
        made = ClusterProfileConfig.model_validate(saved.config or {})
    except ValueError:
        return False
    return made.model_dump() == wanted.model_dump()


@dataclass
class ClusterState:
    """What is already saved for one node: whether its profile is done."""

    node: int
    profile_done: bool

    @property
    def done(self) -> bool:
        return self.profile_done


def cluster_states(
    experiment, run_id: str, nodes: list[int], config: ProfileDescribeConfig
) -> list[ClusterState]:
    """What is already done for each of ``nodes`` under ``config``. Every node is to do when
    ``skip_existing`` is off."""
    from detail.api import cluster_profile

    if not config.skip_existing:
        return [ClusterState(int(n), False) for n in nodes]
    run_dir = experiment.analysis_run_dir(run_id)
    wanted = profile_config(config)
    return [
        ClusterState(
            int(node),
            profile_matches(cluster_profile.load_profile(run_dir, node), wanted),
        )
        for node in nodes
    ]


# --------------------------------------------------------------------------- #
# Placing a job's results beside the run they describe
# --------------------------------------------------------------------------- #
def place(experiment, job_run_id: str, nodes=None) -> dict[str, str]:
    """Put a batch job's results where the single-cluster surfaces keep them: each
    ``profiles/node_<id>/`` of the job's own run directory — its profile and its comparisons
    with other clusters (``against_<id>/``, what Find look-alike clusters makes) — into the
    clustering run's ``profiles/node_<id>/``, where its Interpret tab reads them.
    ``{job path: placed path}``, both relative to their runs, for every file now in place.

    The job writes only into its own run directory — the one directory a remote run's
    copy-back returns — and this is the step that publishes it, run after each cluster on this
    machine and after the copy-back of a remote one. **Idempotent**: a profile is placed when
    the run's is absent or older (an equal one is left alone); so placing twice, or after a
    crash between two files, places nothing twice.

    Raises:
        FileNotFoundError: No such job run, or it names no source run.
    """
    from detail.api import cluster_profile
    from detail.core import atomic

    record = experiment.analysis_record(job_run_id)
    source = record.get("source_run_id") or (record.get("params") or {}).get(
        "source_run_id"
    )
    if not source:
        raise FileNotFoundError(f"{job_run_id} names no clustering run to place into")
    own = experiment.analysis_run_dir(job_run_id)
    target = experiment.analysis_run_dir(source)
    staged = own / cluster_profile.PROFILES_DIRNAME
    wanted = None if nodes is None else {int(n) for n in nodes}
    placed: dict[str, str] = {}
    for folder in sorted(staged.glob("node_*")) if staged.is_dir() else []:
        node = int(folder.name.split("_", 1)[1])
        if wanted is not None and node not in wanted:
            continue
        _place_profile(own, target, node, None, placed, atomic)
        for pair in sorted(folder.glob("against_*")):
            against = int(pair.name.split("_", 1)[1])
            _place_profile(own, target, node, against, placed, atomic)
    return placed


def _place_profile(own: Path, target: Path, node: int, against, placed, atomic) -> None:
    """Place ``node``'s profile (or its comparison with ``against``) when the run's is
    absent or older."""
    from detail.api import cluster_profile

    mine = cluster_profile.load_profile(own, node, against)
    theirs = cluster_profile.load_profile(target, node, against)
    if mine is None or (theirs is not None and theirs.created_at > mine.created_at):
        return
    folder = cluster_profile.profile_dir(own, node, against)
    dest = cluster_profile.profile_dir(target, node, against)
    for name in (cluster_profile.PROFILE_FILENAME, cluster_profile.MEASURES_FILENAME):
        if (folder / name).is_file():
            _place_file(folder / name, dest / name, atomic)
            placed[_rel(folder / name, own)] = _rel(dest / name, target)


def _place_file(src: Path, dst: Path, atomic) -> None:
    data = src.read_bytes()
    if dst.is_file() and dst.read_bytes() == data:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    tmp.write_bytes(data)
    atomic.replace(tmp, dst)


def _rel(path: Path, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


# --------------------------------------------------------------------------- #
# How long it should take
# --------------------------------------------------------------------------- #
@dataclass
class BatchEstimate:
    """How long a batch should take on this machine, and what the figure rests on."""

    n_clusters: int
    n_done: int
    n_profile: int
    profile_s: float  # every profile to do
    profile_basis: str

    @property
    def seconds(self) -> float:
        return self.profile_s

    def text(self) -> str:
        """The lines a surface shows before starting."""
        head = f"{self.n_clusters} cluster" + ("s" if self.n_clusters != 1 else "")
        if self.n_done:
            head += f", {self.n_done} already done (skipped)"
        lines = [head]
        if self.n_profile:
            lines.append(
                f"profiles: {self.n_profile} × {self.profile_basis} = about "
                f"{format_duration(self.profile_s)}"
            )
        return "\n".join(lines)


def profile_seconds(run_dir, config: ClusterProfileConfig) -> tuple[float, str]:
    """What one profile should take: the median of this run's own profiles made with these
    settings, else the time budget (a fixed count has no budget: then unmeasured profiles are
    taken at the default budget)."""
    from detail.api import cluster_profile

    folder = Path(run_dir) / cluster_profile.PROFILES_DIRNAME
    taken = []
    for path in sorted(folder.glob("node_*")) if folder.is_dir() else []:
        try:
            node = int(path.name.split("_", 1)[1])
        except ValueError:
            continue
        saved = cluster_profile.load_profile(run_dir, node)
        if profile_matches(saved, config):
            taken.append(float(saved.elapsed_s))
    if taken:
        median = statistics.median(taken)
        return median, (
            f"{format_duration(median)} (the median of {len(taken)} profile"
            + ("s" if len(taken) != 1 else "")
            + " this run already has)"
        )
    budget = float(config.time_budget_s) + PROFILE_OVERHEAD_S
    if config.n_per_side is not None:
        return budget, (
            f"about {format_duration(budget)} (a fixed count is not timed in advance; "
            "taken at the time budget)"
        )
    return budget, f"{format_duration(budget)} (the time budget)"


def estimate(experiment, run_id: str, config: ProfileDescribeConfig) -> BatchEstimate:
    """How long ``config`` should take over run ``run_id``."""
    run = experiment._clustering_run(run_id)
    view = run.node_view()
    nodes = select_nodes(view, config.clusters, config.node_ids)
    states = cluster_states(experiment, run_id, nodes, config)
    to_profile = [s for s in states if not s.profile_done]
    per_profile, profile_basis = profile_seconds(
        experiment.analysis_run_dir(run_id), profile_config(config)
    )
    return BatchEstimate(
        n_clusters=len(nodes),
        n_done=sum(1 for s in states if s.done),
        n_profile=len(to_profile),
        profile_s=per_profile * len(to_profile),
        profile_basis=profile_basis,
    )
